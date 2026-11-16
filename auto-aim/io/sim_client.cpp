#include "sim_client.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fcntl.h>
#include <regex>
#include <stdexcept>
#include <sys/file.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;

namespace io
{
namespace
{
constexpr auto STATE_MAX_AGE = 250ms;
constexpr std::size_t MAX_JSON_BYTES = 65536;
constexpr std::size_t MAX_IMAGE_BYTES = 64 * 1024 * 1024;

bool valid_channel(const std::string & channel)
{
  static const std::regex pattern("[A-Za-z0-9_-]{1,80}");
  return std::regex_match(channel, pattern);
}
}  // namespace

SimClient::SimClient(const std::string & channel)
{
  if (!valid_channel(channel)) throw std::runtime_error("Invalid simulator channel: " + channel);
  directory_path_ = "/dev/shm/rm_aim_sim_" + channel;
  directory_ = ::open(directory_path_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (directory_ < 0)
    throw std::runtime_error(
      "Cannot open " + directory_path_ + "; start rm_aim_sim with --remote first");

  struct stat info {};
  if (
    ::fstat(directory_, &info) != 0 || info.st_uid != ::getuid() ||
    (info.st_mode & 0777) != 0700) {
    ::close(directory_);
    directory_ = -1;
    throw std::runtime_error("Simulator IPC directory must belong to this user with mode 0700");
  }

  lock_ = ::openat(directory_, "client.lock", O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (lock_ < 0 || ::flock(lock_, LOCK_EX | LOCK_NB) != 0) {
    if (lock_ >= 0) ::close(lock_);
    ::close(directory_);
    lock_ = -1;
    directory_ = -1;
    throw std::runtime_error("Simulator channel already has a client");
  }
  request_token_ = "cool-sentry-" + std::to_string(::getpid());
}

SimClient::~SimClient()
{
  try {
    send({false, false, 0, 0});
  } catch (...) {
  }
  if (lock_ >= 0) ::close(lock_);
  if (directory_ >= 0) ::close(directory_);
}

bool SimClient::refresh_state()
{
  auto value = read_json("state.json");
  if (
    value.value("protocol", "") != "rm_aim_sim/2" || value.value("type", "") != "state")
    return false;

  const double wall_now = std::chrono::duration<double>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
  const double age = wall_now - value.value("wall_unix_sec", 0.0);
  if (!std::isfinite(age) || age < -0.1 || age > 0.5) return false;

  const bool changed =
    state_.empty() || value.value("publisher_id", "") != state_.value("publisher_id", "") ||
    value.value("publication", -1LL) != state_.value("publication", -1LL);
  state_ = std::move(value);
  if (changed) last_state_received_ = std::chrono::steady_clock::now();
  return changed;
}

bool SimClient::state_fresh() const
{
  return !state_.empty() && std::chrono::steady_clock::now() - last_state_received_ <= STATE_MAX_AGE;
}

SimFrame SimClient::next_frame(std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while ((!refresh_state() && !state_fresh()) || state_.empty()) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("Timed out waiting for fresh simulator state");
    std::this_thread::sleep_for(2ms);
  }
  if (!state_.at("ipc").value("enabled", false))
    throw std::runtime_error("Simulator external control is disabled; start it with --remote");

  if (pending_request_.empty()) request_next_frame();
  auto publisher = pending_publisher_;
  auto session = pending_session_;
  auto request_id = pending_request_;

  nlohmann::json frame;
  while (std::chrono::steady_clock::now() < deadline) {
    refresh_state();
    if (!state_fresh()) {
      std::this_thread::sleep_for(2ms);
      continue;
    }
    if (!state_.at("ipc").value("enabled", false)) {
      pending_request_.clear();
      throw std::runtime_error(
        "Simulator external control is disabled; enable 本机自瞄控制 in rm_aim_sim");
    }
    if (state_.value("paused", false)) {
      pending_request_.clear();
      throw std::runtime_error("Simulator is paused; resume rm_aim_sim before starting auto aim");
    }
    if (
      state_.value("publisher_id", "") != publisher ||
      state_.value("session_id", "") != session) {
      pending_request_.clear();
      if (!state_.at("ipc").value("enabled", false) || state_.value("paused", false))
        throw std::runtime_error("Simulator stopped external control while waiting for a frame");
      request_next_frame();
      publisher = pending_publisher_;
      session = pending_session_;
      request_id = pending_request_;
      frame = nlohmann::json::object();
      continue;
    }
    frame = read_json("frame.json");
    if (
      frame.value("protocol", "") == "rm_aim_sim/2" && frame.value("type", "") == "frame" &&
      frame.value("publisher_id", "") == publisher && frame.value("session_id", "") == session &&
      frame.value("request_id", "") == request_id)
      break;
    std::this_thread::sleep_for(2ms);
  }
  if (frame.value("request_id", "") != request_id) {
    pending_request_.clear();
    throw std::runtime_error(
      "Timed out waiting for a simulator camera frame; keep the rm_aim_sim window visible and "
      "not minimized");
  }

  const auto & camera = frame.at("camera");
  const int width = camera.at("width").get<int>();
  const int height = camera.at("height").get<int>();
  const auto encoding = frame.at("encoding").get<std::string>();
  const int channels = encoding == "rgb8" ? 3 : encoding == "rgba8" ? 4 : 0;
  if (!channels || width < 160 || width > 3840 || height < 120 || height > 2160)
    throw std::runtime_error("Invalid simulator image layout");

  const std::string filename = "frame-" + request_id + ".raw";
  const std::size_t bytes = static_cast<std::size_t>(width) * height * channels;
  if (
    bytes > MAX_IMAGE_BYTES || frame.value("frame_file", "") != filename ||
    frame.value("image_bytes", std::size_t{0}) != bytes)
    throw std::runtime_error("Simulator image metadata does not match the requested frame");

  int fd = ::openat(directory_, filename.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) throw std::runtime_error("Simulator frame disappeared before it was read");
  cv::Mat pixels;
  try {
    struct stat info {};
    if (
      ::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size != static_cast<off_t>(bytes))
      throw std::runtime_error("Simulator frame file has an invalid size or type");
    pixels.create(height, width, CV_MAKETYPE(CV_8U, channels));
    read_all(fd, pixels.data, bytes);
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);

  SimFrame result;
  cv::cvtColor(
    pixels, result.bgr, channels == 3 ? cv::COLOR_RGB2BGR : cv::COLOR_RGBA2BGR);
  const auto quaternion = frame.at("gimbal").at("quaternion_wxyz").get<std::vector<double>>();
  if (quaternion.size() != 4) throw std::runtime_error("Invalid simulator quaternion");
  result.gimbal_to_world =
    Eigen::Quaterniond(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
  const double norm = result.gimbal_to_world.norm();
  if (!result.gimbal_to_world.coeffs().allFinite() || norm < 0.8 || norm > 1.2)
    throw std::runtime_error("Invalid simulator quaternion norm");
  result.gimbal_to_world.normalize();
  result.sim_time = frame.at("sim_time").get<double>();
  if (!std::isfinite(result.sim_time) || result.sim_time < 0)
    throw std::runtime_error("Invalid simulator timestamp");
  result.timestamp = std::chrono::steady_clock::time_point(
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(result.sim_time)));
  result.bullet_speed = frame.at("weapon").at("speed").get<double>();
  result.metadata = frame;
  const auto current_context = context(frame);
  result.context_changed = current_context != frame_context_;
  pending_request_.clear();
  refresh_state();
  if (
    state_.value("publisher_id", "") != publisher ||
    state_.value("session_id", "") != session) {
    if (!state_.at("ipc").value("enabled", false) || state_.value("paused", false))
      throw std::runtime_error("Simulator stopped external control after publishing a frame");
    request_next_frame();
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
    if (remaining <= 0ms) throw std::runtime_error("Simulator context kept changing");
    return next_frame(remaining);
  }
  const double frame_age = state_.value("sim_time", result.sim_time) - result.sim_time;
  if (!std::isfinite(frame_age) || frame_age < -0.1 || frame_age > 0.2) {
    if (
      state_fresh() && state_.value("publisher_id", "") == publisher &&
      state_.value("session_id", "") == session && state_.at("ipc").value("enabled", false) &&
      !state_.value("paused", false)) {
      request_next_frame();
      return next_frame(timeout);
    }
    throw std::runtime_error("Simulator returned a stale camera frame");
  }
  frame_context_ = current_context;
  if (
    state_fresh() && state_.value("publisher_id", "") == publisher &&
    state_.value("session_id", "") == session && state_.at("ipc").value("enabled", false) &&
    !state_.value("paused", false))
    request_next_frame();
  return result;
}

void SimClient::request_next_frame()
{
  pending_publisher_ = state_.at("publisher_id").get<std::string>();
  pending_session_ = state_.at("session_id").get<std::string>();
  pending_request_ = request_token_ + "-" + std::to_string(request_count_++);
  write_json(
    "request.json", {{"publisher_id", pending_publisher_}, {"request_id", pending_request_}});
}

void SimClient::send(Command command)
{
  refresh_state();
  if (state_.empty()) return;
  if (!std::isfinite(command.yaw) || !std::isfinite(command.pitch) ||
      !std::isfinite(command.yaw_vel) || !std::isfinite(command.pitch_vel))
    command = {false, false, 0, 0};

  const auto session = state_.at("session_id").get<std::string>();
  if (sequence_session_ != session) {
    sequence_session_ = session;
    sequence_ = 0;
  }
  const auto next_sequence =
    state_.at("ipc").value("last_seq", static_cast<std::int64_t>(-1)) + 1;
  sequence_ = std::max(sequence_, next_sequence);
  const bool available =
    state_fresh() && state_.at("ipc").value("enabled", false) && !state_.value("paused", false);
  command.control = command.control && available;
  command.shoot = command.shoot && command.control;
  write_json(
    "command.json",
    {{"protocol", "rm_aim_sim/2"},
     {"type", "command"},
     {"publisher_id", state_.at("publisher_id")},
     {"session_id", state_.at("session_id")},
     {"seq", sequence_++},
     {"control", command.control},
     {"fire", command.shoot},
     {"yaw", command.yaw},
     {"pitch", command.pitch},
     {"yaw_vel", command.yaw_vel},
     {"pitch_vel", command.pitch_vel},
     {"yaw_acc", 0.0},
     {"pitch_acc", 0.0}});
}

const nlohmann::json & SimClient::state() const { return state_; }

double SimClient::state_sim_time() const { return state_.value("sim_time", 0.0); }

nlohmann::json SimClient::read_json(const char * filename) const
{
  int fd = ::openat(directory_, filename, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return nlohmann::json::object();
  std::string data;
  try {
    struct stat info {};
    if (
      ::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
      static_cast<std::size_t>(info.st_size) > MAX_JSON_BYTES)
      throw std::runtime_error(std::string("Invalid simulator JSON file: ") + filename);
    data.resize(info.st_size);
    read_all(fd, data.data(), data.size());
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  auto value = nlohmann::json::parse(data, nullptr, false);
  return value.is_object() ? value : nlohmann::json::object();
}

void SimClient::write_json(const char * filename, const nlohmann::json & value) const
{
  const std::string temporary =
    std::string(filename) + ".client-" + std::to_string(::getpid()) + ".tmp";
  int fd = ::openat(
    directory_, temporary.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) throw std::runtime_error(std::string("Cannot write simulator ") + filename);
  try {
    const auto data = value.dump();
    std::size_t written = 0;
    while (written < data.size()) {
      const auto count = ::write(fd, data.data() + written, data.size() - written);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) throw std::runtime_error("Incomplete simulator JSON write");
      written += static_cast<std::size_t>(count);
    }
  } catch (...) {
    ::close(fd);
    ::unlinkat(directory_, temporary.c_str(), 0);
    throw;
  }
  ::close(fd);
  if (::renameat(directory_, temporary.c_str(), directory_, filename) != 0) {
    ::unlinkat(directory_, temporary.c_str(), 0);
    throw std::runtime_error(std::string("Cannot publish simulator ") + filename);
  }
}

std::string SimClient::context(const nlohmann::json & packet)
{
  const auto & camera = packet.at("camera");
  return nlohmann::json::array(
           {packet.at("publisher_id"), packet.at("session_id"), packet.at("appearance"),
            camera.at("width"), camera.at("height"), camera.at("K"), camera.at("distortion"),
            camera.at("R_camera2gimbal"), camera.at("t_camera2gimbal")})
    .dump();
}

void SimClient::read_all(int fd, void * output, std::size_t bytes)
{
  auto * data = static_cast<char *>(output);
  while (bytes) {
    const auto count = ::read(fd, data, bytes);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("Incomplete simulator file read");
    data += count;
    bytes -= static_cast<std::size_t>(count);
  }
}

}  // namespace io
