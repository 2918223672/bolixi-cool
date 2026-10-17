#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sstream>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{display d      | true              | 显示回放窗口      }"
  "{metrics-out m  |                   | 汇总指标JSON路径  }"
  "{frames-out     |                   | 逐帧指标JSONL路径 }"
  "{bullet-speed   | 27                | 旧录像默认弹速m/s }"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径}";

namespace
{
struct RunningStats
{
  std::size_t count = 0;
  double mean = 0;
  double squared_delta = 0;
  double squared_sum = 0;

  void add(double value)
  {
    ++count;
    const double delta = value - mean;
    mean += delta / static_cast<double>(count);
    squared_delta += delta * (value - mean);
    squared_sum += value * value;
  }

  double stddev() const
  {
    return count > 0 ? std::sqrt(squared_delta / static_cast<double>(count)) : 0;
  }

  double rms() const { return count > 0 ? std::sqrt(squared_sum / static_cast<double>(count)) : 0; }
};

struct ReplaySample
{
  double time = 0;
  Eigen::Quaterniond quaternion{1, 0, 0, 0};
  double bullet_speed = 27;
};

bool read_sample(std::ifstream & input, double default_bullet_speed, ReplaySample & sample)
{
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream stream(line);
    double w, x, y, z;
    if (!(stream >> sample.time >> w >> x >> y >> z)) return false;
    sample.quaternion = Eigen::Quaterniond(w, x, y, z);
    sample.bullet_speed = default_bullet_speed;
    stream >> sample.bullet_speed;
    return true;
  }
  return false;
}

double reprojection_rmse(const auto_aim::Solver & solver, const auto_aim::Armor & armor)
{
  const auto points =
    solver.reproject_armor(armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
  if (points.size() != armor.points.size() || points.empty()) return 0;
  double squared_error = 0;
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double error = cv::norm(points[i] - armor.points[i]);
    squared_error += error * error;
  }
  return std::sqrt(squared_error / static_cast<double>(points.size()));
}
}  // namespace

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto display = cli.get<bool>("display");
  auto metrics_path = cli.get<std::string>("metrics-out");
  auto frames_path = cli.get<std::string>("frames-out");
  auto default_bullet_speed = cli.get<double>("bullet-speed");

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);
  if (!video.isOpened() || !text.is_open()) {
    tools::logger()->error("Unable to open replay files: {}.avi/.txt", input_path);
    return 2;
  }
  std::ofstream frames_output;
  if (!frames_path.empty()) frames_output.open(frames_path);

  auto_aim::YOLO yolo(config_path, display);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img, drawing;
  auto t0 = std::chrono::steady_clock::now();

  io::Command last_command{false, false, 0, 0};
  std::optional<double> last_armor_yaw;
  std::optional<int> last_armor_frame;
  std::optional<auto_aim::ArmorName> last_armor_name;
  std::optional<auto_aim::ArmorType> last_armor_type;
  std::optional<double> last_command_yaw;
  RunningStats processing_ms;
  RunningStats reprojection_error;
  RunningStats yaw_correction_deg;
  RunningStats armor_yaw_step_deg;
  RunningStats command_yaw_step_deg;
  RunningStats nis;
  std::size_t frames = 0;
  std::size_t detected_frames = 0;
  std::size_t tracked_frames = 0;
  std::size_t controlled_frames = 0;
  std::size_t nis_rejections = 0;
  std::size_t tracker_loss_events = 0;
  std::string previous_tracker_state = tracker.state();

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    ReplaySample skipped;
    if (!read_sample(text, default_bullet_speed, skipped)) break;
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    ReplaySample sample;
    if (!read_sample(text, default_bullet_speed, sample)) {
      tools::logger()->warn("Replay metadata ended at frame {}", frame_count);
      break;
    }
    auto timestamp = t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(sample.time));

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world(sample.quaternion);

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    ++frames;
    if (!armors.empty()) ++detected_frames;

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, sample.bullet_speed, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    const double frame_processing_ms = tools::delta_time(finish, yolo_start) * 1e3;
    processing_ms.add(frame_processing_ms);
    if (!targets.empty()) ++tracked_frames;
    if (command.control) ++controlled_frames;
    if (previous_tracker_state != "lost" && tracker.state() == "lost") ++tracker_loss_events;
    previous_tracker_state = tracker.state();

    bool has_measurement = false;
    if (!armors.empty() && !targets.empty()) {
      const auto & target = targets.front();
      const auto armor = std::find_if(armors.begin(), armors.end(), [&](const auto & candidate) {
        return candidate.name == target.name && candidate.type == target.armor_type;
      });
      if (armor != armors.end()) {
        has_measurement = true;
        reprojection_error.add(reprojection_rmse(solver, *armor));
        yaw_correction_deg.add(tools::limit_rad(armor->ypr_in_world[0] - armor->yaw_raw) * 57.3);
        if (
          last_armor_yaw.has_value() && last_armor_frame == frame_count - 1 &&
          last_armor_name == armor->name && last_armor_type == armor->type) {
          armor_yaw_step_deg.add(tools::limit_rad(armor->ypr_in_world[0] - *last_armor_yaw) * 57.3);
        }
        last_armor_yaw = armor->ypr_in_world[0];
        last_armor_frame = frame_count;
        last_armor_name = armor->name;
        last_armor_type = armor->type;
      }
    }

    if (command.control) {
      if (last_command_yaw.has_value()) {
        command_yaw_step_deg.add(tools::limit_rad(command.yaw - *last_command_yaw) * 57.3);
      }
      last_command_yaw = command.yaw;
    }

    if (has_measurement && !targets.empty()) {
      const auto & ekf = targets.front().ekf();
      if (ekf.last_nis_threshold > 0) {
        nis.add(ekf.last_nis);
        if (ekf.last_nis > ekf.last_nis_threshold) ++nis_rejections;
      }
    }
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion gimbal_q = sample.quaternion;
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    auto yaw = tools::eulers(sample.quaternion, 2, 1, 0)[0];
    data["gimbal_yaw"] = yaw * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nis_threshold"] = target.ekf().data.at("nis_threshold");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    plotter.plot(data);
    if (frames_output.is_open()) {
      data["frame"] = frame_count;
      data["time"] = sample.time;
      data["bullet_speed"] = sample.bullet_speed;
      data["tracker_state"] = tracker.state();
      data["processing_ms"] = frame_processing_ms;
      frames_output << data.dump() << '\n';
    }

    if (display) {
      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("reprojection", img);
      if (cv::waitKey(30) == 'q') break;
    }
  }

  const auto ratio = [frames](std::size_t count) {
    return frames > 0 ? static_cast<double>(count) / static_cast<double>(frames) : 0.0;
  };
  nlohmann::json summary{
    {"schema_version", 1},
    {"dataset", input_path},
    {"config", config_path},
    {"frames", frames},
    {"detected_frames", detected_frames},
    {"tracked_frames", tracked_frames},
    {"controlled_frames", controlled_frames},
    {"detection_rate", ratio(detected_frames)},
    {"tracking_rate", ratio(tracked_frames)},
    {"control_rate", ratio(controlled_frames)},
    {"tracker_loss_events", tracker_loss_events},
    {"processing_mean_ms", processing_ms.mean},
    {"reprojection_samples", reprojection_error.count},
    {"reprojection_rmse_mean_px", reprojection_error.mean},
    {"reprojection_rmse_std_px", reprojection_error.stddev()},
    {"yaw_correction_mean_deg", yaw_correction_deg.mean},
    {"yaw_correction_std_deg", yaw_correction_deg.stddev()},
    {"armor_yaw_step_rms_deg", armor_yaw_step_deg.rms()},
    {"command_yaw_step_rms_deg", command_yaw_step_deg.rms()},
    {"nis_updates", nis.count},
    {"nis_mean", nis.mean},
    {"nis_rejection_rate",
     nis.count > 0 ? static_cast<double>(nis_rejections) / static_cast<double>(nis.count) : 0.0}};

  if (!metrics_path.empty()) {
    std::ofstream metrics_output(metrics_path);
    metrics_output << summary.dump(2) << '\n';
  }
  fmt::print("METRICS {}\n", summary.dump());

  return 0;
}
