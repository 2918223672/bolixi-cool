#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
//planner / aimer / shooter: 负责根据目标的运动轨迹，预测未来的位置，并决定云台该如何转动，以及何时触发发射。
namespace auto_aim
{
Shooter::Shooter(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  spin_tolerance_ = yaml["spin_tolerance"].as<double>(0.0) / 57.3;
  spin_moving_tolerance_ = yaml["spin_moving_tolerance"].as<double>(0.0) / 57.3;
  spin_speed_threshold_ = yaml["spin_speed_threshold"].as<double>(0.0);
  auto_fire_ = yaml["auto_fire"].as<bool>();
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  if (!command.control || targets.empty() || !auto_fire_) return false;

  auto target_x = targets.front().ekf_x()[0];
  auto target_y = targets.front().ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;
  if (std::hypot(targets.front().ekf_x()[1], targets.front().ekf_x()[3]) > 0.8)
    observed_translation_ = true;
  if (spin_tolerance_ > 0 &&
      std::abs(targets.front().ekf_x()[7]) >= spin_speed_threshold_)
    tolerance = std::min(
      tolerance, observed_translation_ && spin_moving_tolerance_ > 0 ?
                   spin_moving_tolerance_ : spin_tolerance_);
  if (
    aimer.fire_command.control &&
    std::abs(tools::limit_rad(gimbal_pos[0] - aimer.fire_command.yaw)) < tolerance &&
    std::abs(tools::limit_rad(gimbal_pos[1] - aimer.fire_command.pitch)) < tolerance &&
    aimer.debug_aim_point.valid) {
    return true;
  }

  return false;
}

}  // namespace auto_aim
