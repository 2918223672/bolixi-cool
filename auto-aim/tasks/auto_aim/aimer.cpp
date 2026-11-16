#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
//planner / aimer / shooter: 负责根据目标的运动轨迹，预测未来的位置，并决定云台该如何转动，以及何时触发发射。
namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree to rad
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  if (yaml["drag_k"]) drag_k_ = yaml["drag_k"].as<double>();
  spin_center_aim_ = yaml["spin_center_aim"].as<bool>(false);
  spin_use_both_heights_ = yaml["spin_use_both_heights"].as<bool>(false);
  spin_speed_threshold_ = yaml["spin_speed_threshold"].as<double>(0.0);
  spin_fire_half_width_ = yaml["spin_fire_half_width"].as<double>(0.0);
  spin_moving_fire_half_width_ =
    yaml["spin_moving_fire_half_width"].as<double>(spin_fire_half_width_);
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  fire_command = {false, false, 0, 0};
  debug_prediction = {};
  if (targets.empty()) {
    return {false, false, 0, 0};
  }
  auto target = targets.front();
  if (std::hypot(target.ekf_x()[1], target.ekf_x()[3]) > 0.8)
    observed_translation_ = true;

  double delay_time =
    std::abs(target.ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (bullet_speed < 14) bullet_speed = 23;

  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  else {
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // tools::logger()->info("dt is {:.4f} second", dt);
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  if (spin_center_aim_ && std::abs(target.ekf_x()[7]) >= spin_speed_threshold_) {
    const bool use_both_heights = spin_use_both_heights_;
    const auto initial_state = target.ekf_x();
    const double initial_yaw = std::atan2(initial_state[2], initial_state[0]);
    const double initial_x = initial_state[0] - initial_state[8] * std::cos(initial_yaw);
    const double initial_y = initial_state[2] - initial_state[8] * std::sin(initial_yaw);
    tools::Trajectory initial_trajectory(
      bullet_speed, std::hypot(initial_x, initial_y), initial_state[4], drag_k_);
    if (initial_trajectory.unsolvable) return {false, false, 0, 0};

    target.predict(future + std::chrono::microseconds(
                              static_cast<int64_t>(initial_trajectory.fly_time * 1e6)));
    const auto state = target.ekf_x();
    const double center_yaw = std::atan2(state[2], state[0]);
    const double front_x = state[0] - state[8] * std::cos(center_yaw);
    const double front_y = state[2] - state[8] * std::sin(center_yaw);
    const auto armors = target.armor_xyza_list();
    double smallest_angle = 1e10;
    Eigen::Vector4d closest = armors.front();
    for (const auto & armor : armors) {
      if (!use_both_heights && std::abs(armor[2] - state[4]) > 0.01) continue;
      const double angle = std::abs(tools::limit_rad(armor[3] - center_yaw));
      if (angle < smallest_angle) {
        smallest_angle = angle;
        closest = armor;
      }
    }
    const double lateral_error =
      std::abs(closest[0] * std::sin(center_yaw) - closest[1] * std::cos(center_yaw));
    const double fire_half_width =
      observed_translation_ ? spin_moving_fire_half_width_ : spin_fire_half_width_;
    debug_aim_point = {smallest_angle < 0.5 && lateral_error <= fire_half_width,
                       closest};
    tools::Trajectory trajectory(
      bullet_speed, std::hypot(front_x, front_y), closest[2], drag_k_);
    if (trajectory.unsolvable) return {false, false, 0, 0};
    fire_command = {
      true, false, center_yaw + yaw_offset_, -(trajectory.pitch + pitch_offset_)};
    return fire_command;
  }

  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    // tools::logger()->debug("Invalid aim_point0.");
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2], drag_k_);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;

  for (int iter = 0; iter < 10; ++iter) {
    // 预测目标在 future + prev_fly_time 时刻的位置
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    Target iteration_target = target;
    iteration_target.predict(predict_time);

    // 计算瞄准点
    auto aim_point = choose_aim_point(iteration_target);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    // 计算新弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z(), drag_k_);

    // 检查弹道是否可解
    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // Record the exact time used for this prediction, including the last iteration.
    const bool converged = std::abs(current_traj.fly_time - prev_fly_time) < 0.001;
    debug_prediction = {true, future, predict_time, current_traj.fly_time, iter + 1, converged};
    // 检查收敛条件
    if (converged) {
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);  //世界坐标系下pitch向上为负
  fire_command = {true, false, yaw, pitch};
  return fire_command;
}

AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 不考虑小陀螺
  if (std::abs(target.ekf_x()[8]) <= 2) {
    // 选择在可射击范围内的装甲板
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    // 绝无可能
    if (id_list.empty()) {
      tools::logger()->warn("Empty id list!");
      return {false, armor_xyza_list[0]};
    }

    // 锁定模式：防止在两个都呈45度的装甲板之间来回切换
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];

      // 未处于锁定模式时，选择delta_angle绝对值较小的装甲板，进入锁定模式
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个装甲板在可射击范围内时，退出锁定模式
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  const double coming_angle = comming_angle_;
  const double leaving_angle = leaving_angle_;

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
