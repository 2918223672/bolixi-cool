#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

struct AimPoint
{
  bool valid;
  Eigen::Vector4d xyza;
  int armor_id = -1;
};

// Diagnostics for the iterative armor-center prediction; never used by control logic.
struct AimPredictionDebug
{
  bool valid = false;
  std::chrono::steady_clock::time_point reference_timestamp{};
  std::chrono::steady_clock::time_point prediction_timestamp{};
  double fly_time = 0;
  int iterations = 0;
  bool converged = false;
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  AimPredictionDebug debug_prediction;
  io::Command fire_command{false, false, 0, 0};
  explicit Aimer(const std::string & config_path);
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

private:
  double yaw_offset_;
  double pitch_offset_;
  double comming_angle_;
  double leaving_angle_;
  double lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;
  double drag_k_ = 0.0;
  bool spin_center_aim_ = false;
  bool spin_use_both_heights_ = false;
  bool observed_translation_ = false;
  double spin_moving_fire_half_width_ = 0;
  double spin_speed_threshold_ = 0;
  double spin_fire_half_width_ = 0;
  AimPoint choose_aim_point(const Target & target);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
