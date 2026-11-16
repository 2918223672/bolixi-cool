#ifndef IO__SIM_CLIENT_HPP
#define IO__SIM_CLIENT_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <string>

#include "io/command.hpp"

namespace io
{

struct SimFrame
{
  cv::Mat bgr;
  Eigen::Quaterniond gimbal_to_world{1, 0, 0, 0};
  std::chrono::steady_clock::time_point timestamp;
  double sim_time = 0;
  double bullet_speed = 0;
  bool context_changed = false;
  nlohmann::json metadata;
};

class SimClient
{
public:
  explicit SimClient(const std::string & channel);
  ~SimClient();
  SimClient(const SimClient &) = delete;
  SimClient & operator=(const SimClient &) = delete;

  SimFrame next_frame(std::chrono::milliseconds timeout = std::chrono::seconds(10));
  bool refresh_state();
  void send(Command command);

  const nlohmann::json & state() const;
  double state_sim_time() const;

private:
  int directory_ = -1;
  int lock_ = -1;
  nlohmann::json state_ = nlohmann::json::object();
  std::string directory_path_;
  std::string sequence_session_;
  std::string frame_context_;
  std::string request_token_;
  std::string pending_request_;
  std::string pending_publisher_;
  std::string pending_session_;
  std::int64_t sequence_ = 0;
  std::uint64_t request_count_ = 0;
  std::chrono::steady_clock::time_point last_state_received_{};

  bool state_fresh() const;
  void request_next_frame();
  nlohmann::json read_json(const char * filename) const;
  void write_json(const char * filename, const nlohmann::json & value) const;
  static std::string context(const nlohmann::json & packet);
  static void read_all(int fd, void * output, std::size_t bytes);
};

}  // namespace io

#endif  // IO__SIM_CLIENT_HPP
