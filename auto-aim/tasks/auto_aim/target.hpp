#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  bool update(const Armor & armor);
  void set_spatial_association(
    double max_error, double relative_error = 0.0, double max_yaw_error = 0.0);
  void set_max_angular_speed(double max_speed);
  void set_angular_acceleration_variance(double variance);
  void set_armor_yaw_noise(double base_variance, double distance_scale);
  void set_imm_parameters(double mode_time_constant, double cv_acceleration_variance,
                          double ca_jerk_variance);
  void fix_normal_geometry(double radius, double radius_difference, double height_difference,
                           double center_z, bool initial_armor_is_low);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  Eigen::Vector2d armor_velocity(int id) const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;
  bool fixed_geometry_ = false;
  double fixed_radius_ = 0;
  double fixed_radius_difference_ = 0;
  double fixed_height_difference_ = 0;
  double fixed_center_z_ = 0;
  double max_spatial_error_ = 0;
  double relative_spatial_error_ = 0;
  double max_association_yaw_error_ = 0;
  double max_angular_speed_ = 0;
  double armor_yaw_noise_base_ = 0.09;
  double armor_yaw_distance_scale_ = -1.0;

  tools::ExtendedKalmanFilter ekf_;
  tools::ExtendedKalmanFilter cv_;
  tools::ExtendedKalmanFilter ca_;
  Eigen::Vector2d model_probability_{0.9, 0.1};
  double imm_mode_time_constant_ = 0.35;
  double cv_acceleration_variance_ = 20.0;
  double ca_jerk_variance_ = 300.0;
  double angular_acceleration_variance_ = 400.0;
  std::chrono::steady_clock::time_point t_;

  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle
  void apply_fixed_geometry();
  void predict_model(tools::ExtendedKalmanFilter & filter, double dt, bool constant_acceleration);
  void update_model(tools::ExtendedKalmanFilter & filter, const Armor & armor, int id);
  void fuse_models();

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
