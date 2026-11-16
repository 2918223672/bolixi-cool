#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>
#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
//(追踪器): 负责记住目标，处理目标的短暂丢失。
namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now())
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  max_frame_gap_ = yaml["max_frame_gap"].as<double>(0.1);
  max_angular_speed_ = yaml["max_angular_speed"].as<double>(0.0);
  const auto ekf = yaml["ekf"];
  if (ekf) {
    imm_mode_time_constant_ = ekf["imm_mode_time_constant"].as<double>(0.35);
    cv_acceleration_variance_ = ekf["imm_cv_acceleration_variance"].as<double>(20.0);
    ca_jerk_variance_ = ekf["imm_ca_jerk_variance"].as<double>(300.0);
    angular_acceleration_variance_ = ekf["angular_acceleration"].as<double>(400.0);
    armor_yaw_noise_base_ = ekf["armor_yaw_noise_base"].as<double>(0.09);
    if (ekf["armor_yaw_noise_base"] || ekf["armor_yaw_distance_scale"])
      armor_yaw_distance_scale_ = ekf["armor_yaw_distance_scale"].as<double>(0.0);
  }
  if (ekf && ekf["use_spatial_association"].as<bool>(false)) {
    max_spatial_error_ = ekf["max_spatial_error"].as<double>(0.25);
    relative_spatial_error_ = ekf["relative_spatial_error"].as<double>(0.0);
    max_association_yaw_error_ = ekf["max_association_yaw_error"].as<double>(0.0);
  }
  const auto sim = yaml["sim"];
  if (sim) {
    const auto forced_name = sim["force_armor_name"].as<std::string>("");
    const auto found = std::find(ARMOR_NAMES.begin(), ARMOR_NAMES.end(), forced_name);
    if (found != ARMOR_NAMES.end())
      forced_armor_name_ = static_cast<ArmorName>(std::distance(ARMOR_NAMES.begin(), found));
    max_coast_image_error_px_ = sim["max_coast_image_error_px"].as<double>(0.0);
    max_spin_coast_frames_ = sim["max_spin_coast_frames"].as<int>(0);
  }
  const auto geometry = yaml["normal_target_geometry"];
  if (geometry && geometry["fixed"].as<bool>(false)) {
    fixed_normal_geometry_ = true;
    normal_radius_ = geometry["radius"].as<double>();
    normal_radius_difference_ = geometry["radius_difference"].as<double>();
    normal_height_difference_ = geometry["height_difference"].as<double>();
    normal_center_z_ = geometry["reference_center_z"].as<double>();
    max_armor_height_error_ = geometry["max_armor_height_error"].as<double>(0.0);
  }
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
}

std::string Tracker::state() const { return state_; }

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > max_frame_gap_) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤掉非我方装甲板
  armors.remove_if([&](const auto_aim::Armor & a) {
    return a.color != enemy_color_ || a.name == ArmorName::outpost;
  });
  if (forced_armor_name_ != ArmorName::not_armor)
    for (auto & armor : armors)
      if (armor.type == ArmorType::small) armor.name = forced_armor_name_;

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    const cv::Point2f img_center(0.5f, 0.5f);
    auto distance_1 = cv::norm(a.center_norm - img_center);
    auto distance_2 = cv::norm(b.center_norm - img_center);
    return distance_1 < distance_2;
  });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  std::list<Target> targets = {target_};
  return targets;
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto candidate = armors.begin();
  for (; candidate != armors.end(); ++candidate) {
    if (solver_.solve(*candidate) && valid_observation(*candidate)) break;
  }
  if (candidate == armors.end()) return false;
  auto & armor = *candidate;

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
    if (
      fixed_normal_geometry_ &&
      (armor.name == ArmorName::three || armor.name == ArmorName::four ||
       armor.name == ArmorName::five))
      target_.fix_normal_geometry(
        normal_radius_, normal_radius_difference_, normal_height_difference_, normal_center_z_,
        std::abs(armor.xyz_in_world.z() - normal_center_z_ - normal_height_difference_) <
          std::abs(armor.xyz_in_world.z() - normal_center_z_));
  }

  target_.set_spatial_association(
    max_spatial_error_, relative_spatial_error_, max_association_yaw_error_);
  target_.set_max_angular_speed(max_angular_speed_);
  target_.set_angular_acceleration_variance(angular_acceleration_variance_);
  if (armor_yaw_distance_scale_ >= 0.0)
    target_.set_armor_yaw_noise(armor_yaw_noise_base_, armor_yaw_distance_scale_);
  target_.set_imm_parameters(
    imm_mode_time_constant_, cv_acceleration_variance_, ca_jerk_variance_);
  invalid_coast_count_ = 0;

  return true;
}

bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  int found_count = 0;
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
  }

  if (found_count == 0) return false;

  bool solved_any = false;
  bool visually_consistent = false;
  const auto predicted_armors = target_.armor_xyza_list();
  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
    )
      continue;

    if (max_coast_image_error_px_ > 0) {
      for (const auto & predicted : predicted_armors) {
        const auto projected = solver_.reproject_armor(
          predicted.head(3), predicted[3], armor.type, armor.name);
        if (projected.size() != 4) continue;
        cv::Point2f center(0, 0);
        for (const auto & point : projected) center += point;
        center *= 0.25f;
        if (cv::norm(center - armor.center) <= max_coast_image_error_px_)
          visually_consistent = true;
      }
    }

    if (!solver_.solve(armor) || !valid_observation(armor)) continue;

    solved_any = target_.update(armor) || solved_any;
  }

  if (solved_any) {
    invalid_coast_count_ = 0;
    return true;
  }
  if (!visually_consistent) return false;
  const auto state = target_.ekf_x();
  const bool nearly_stationary = std::hypot(state[1], state[3]) < 0.15 &&
                                std::abs(state[7]) < 0.5;
  if (nearly_stationary || invalid_coast_count_ < max_spin_coast_frames_) {
    ++invalid_coast_count_;
    return true;
  }
  return false;
}

bool Tracker::valid_observation(const Armor & armor) const
{
  if (!fixed_normal_geometry_ || max_armor_height_error_ <= 0 ||
      (armor.name != ArmorName::three && armor.name != ArmorName::four &&
       armor.name != ArmorName::five))
    return true;

  const auto height = armor.xyz_in_world.z();
  const auto high_error = std::abs(height - normal_center_z_);
  const auto low_error = std::abs(height - normal_center_z_ - normal_height_difference_);
  return std::min(high_error, low_error) <= max_armor_height_error_;
}

}  // namespace auto_aim
