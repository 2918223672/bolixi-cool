#include "target.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h ax ay
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(13, 13);
  P0.topLeftCorner(11, 11) = P0_dig.asDiagonal();
  P0(11, 11) = P0(12, 12) = 9.0;

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  cv_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ca_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h, 0, 0}};
  Eigen::VectorXd P0_dig = Eigen::VectorXd::Zero(13);
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  cv_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  ca_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (!(dt > 0.0) || !std::isfinite(dt)) return;
  const double cv_to_ca = -std::expm1(-dt / imm_mode_time_constant_);
  const double ca_to_cv = -std::expm1(-dt / imm_mode_time_constant_);
  const Eigen::Vector2d predicted_probability(
    model_probability_[0] * (1.0 - cv_to_ca) + model_probability_[1] * ca_to_cv,
    model_probability_[0] * cv_to_ca + model_probability_[1] * (1.0 - ca_to_cv));
  const Eigen::VectorXd cv_x = cv_.x;
  const Eigen::VectorXd ca_x = ca_.x;
  const Eigen::MatrixXd cv_P = cv_.P;
  const Eigen::MatrixXd ca_P = ca_.P;
  auto mix_into = [&](tools::ExtendedKalmanFilter & destination, int destination_mode) {
    const double normalizer = predicted_probability[destination_mode];
    const double cv_transition = destination_mode == 0 ? 1.0 - cv_to_ca : cv_to_ca;
    const double ca_transition = destination_mode == 0 ? ca_to_cv : 1.0 - ca_to_cv;
    const double cv_weight = model_probability_[0] * cv_transition / normalizer;
    const double ca_weight = model_probability_[1] * ca_transition / normalizer;
    Eigen::VectorXd mean = cv_weight * cv_x + ca_weight * ca_x;
    mean[6] = tools::limit_rad(cv_x[6] + ca_weight * tools::limit_rad(ca_x[6] - cv_x[6]));
    Eigen::VectorXd cv_delta = cv_x - mean;
    Eigen::VectorXd ca_delta = ca_x - mean;
    cv_delta[6] = tools::limit_rad(cv_delta[6]);
    ca_delta[6] = tools::limit_rad(ca_delta[6]);
    Eigen::MatrixXd covariance =
      cv_weight * (cv_P + cv_delta * cv_delta.transpose()) +
      ca_weight * (ca_P + ca_delta * ca_delta.transpose());
    destination.x = mean;
    destination.P = covariance;
  };
  mix_into(cv_, 0);
  mix_into(ca_, 1);
  predict_model(cv_, dt, false);
  predict_model(ca_, dt, true);
  model_probability_ = predicted_probability;
  fuse_models();
  if (max_angular_speed_ > 0)
    ekf_.x[7] = std::clamp(ekf_.x[7], -max_angular_speed_, max_angular_speed_);
  apply_fixed_geometry();
}

void Target::predict_model(tools::ExtendedKalmanFilter & filter, double dt, bool ca_model)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(13, 13);
  F(0, 1) = F(2, 3) = F(4, 5) = F(6, 7) = dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(13, 13);
  const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
  constexpr double angular_acceleration_variance = 400.0;
  if (ca_model) {
    F(0, 11) = F(2, 12) = 0.5 * dt2;
    F(1, 11) = F(3, 12) = dt;
    for (const auto & axis : {std::array<int, 3>{0, 1, 11}, std::array<int, 3>{2, 3, 12}}) {
      const int p = axis[0], v = axis[1], a = axis[2];
      Q(p, p) = ca_jerk_variance_ * dt5 / 20.0;
      Q(p, v) = Q(v, p) = ca_jerk_variance_ * dt4 / 8.0;
      Q(p, a) = Q(a, p) = ca_jerk_variance_ * dt3 / 6.0;
      Q(v, v) = ca_jerk_variance_ * dt3 / 3.0;
      Q(v, a) = Q(a, v) = ca_jerk_variance_ * dt2 / 2.0;
      Q(a, a) = ca_jerk_variance_ * dt;
    }
  } else {
    F(11, 11) = F(12, 12) = 0.0;
    for (const auto & axis : {std::array<int, 2>{0, 1}, std::array<int, 2>{2, 3}}) {
      const int p = axis[0], v = axis[1];
      Q(p, p) = cv_acceleration_variance_ * dt4 / 4.0;
      Q(p, v) = Q(v, p) = cv_acceleration_variance_ * dt3 / 2.0;
      Q(v, v) = cv_acceleration_variance_ * dt2;
    }
    Q(11, 11) = Q(12, 12) = 1e-9;
  }
  Q(6, 6) = angular_acceleration_variance * dt4 / 4.0;
  Q(6, 7) = Q(7, 6) = angular_acceleration_variance * dt3 / 2.0;
  Q(7, 7) = angular_acceleration_variance * dt2;
  filter.predict(F, Q, [&](const Eigen::VectorXd & state) {
    Eigen::VectorXd predicted = F * state;
    predicted[6] = tools::limit_rad(predicted[6]);
    return predicted;
  });
  if (max_angular_speed_ > 0)
    filter.x[7] = std::clamp(filter.x[7], -max_angular_speed_, max_angular_speed_);
}

void Target::fuse_models()
{
  const double sum = model_probability_.sum();
  if (!(sum > 0.0) || !std::isfinite(sum)) model_probability_ << 0.9, 0.1;
  else model_probability_ /= sum;
  ekf_.x = model_probability_[0] * cv_.x + model_probability_[1] * ca_.x;
  ekf_.x[6] = tools::limit_rad(cv_.x[6] + model_probability_[1] *
    tools::limit_rad(ca_.x[6] - cv_.x[6]));
  ekf_.P = Eigen::MatrixXd::Zero(13, 13);
  for (const auto & source : {std::pair<const tools::ExtendedKalmanFilter *, double>{&cv_, model_probability_[0]},
                              {&ca_, model_probability_[1]}}) {
    Eigen::VectorXd delta = source.first->x - ekf_.x;
    delta[6] = tools::limit_rad(delta[6]);
    ekf_.P.noalias() += source.second * (source.first->P + delta * delta.transpose());
  }
  ekf_.data = model_probability_[0] >= model_probability_[1] ? cv_.data : ca_.data;
  ekf_.data["imm_cv_probability"] = model_probability_[0];
  ekf_.data["imm_ca_probability"] = model_probability_[1];
  ekf_.last_nis = model_probability_[0] * cv_.last_nis + model_probability_[1] * ca_.last_nis;
}

void Target::fix_normal_geometry(
  double radius, double radius_difference, double height_difference, double center_z,
  bool initial_armor_is_low)
{
  const double radius_delta = radius - ekf_.x[8];
  ekf_.x[0] += radius_delta * std::cos(ekf_.x[6]);
  ekf_.x[2] += radius_delta * std::sin(ekf_.x[6]);
  if (initial_armor_is_low) {
    ekf_.x[6] = tools::limit_rad(ekf_.x[6] - CV_PI / 2);
    last_id = 1;
    jumped = true;
  }
  fixed_geometry_ = true;
  fixed_radius_ = radius;
  fixed_radius_difference_ = radius_difference;
  fixed_height_difference_ = height_difference;
  fixed_center_z_ = center_z;
  cv_.x = ca_.x = ekf_.x;
  cv_.P = ca_.P = ekf_.P;
  model_probability_ << 0.9, 0.1;
  apply_fixed_geometry();
}

void Target::apply_fixed_geometry()
{
  if (!fixed_geometry_) return;

  for (auto * filter : {&ekf_, &cv_, &ca_}) {
    for (const auto index : {4, 5, 8, 9, 10}) {
      filter->P.row(index).setZero();
      filter->P.col(index).setZero();
    }
    filter->x[4] = fixed_center_z_;
    filter->x[5] = 0;
    filter->x[8] = fixed_radius_;
    filter->x[9] = fixed_radius_difference_;
    filter->x[10] = fixed_height_difference_;
  }
}

void Target::set_spatial_association(double max_error)
{
  max_spatial_error_ = max_error;
}

void Target::set_max_angular_speed(double max_speed)
{
  max_angular_speed_ = max_speed;
}

void Target::set_imm_parameters(
  double mode_time_constant, double cv_acceleration_variance, double ca_jerk_variance)
{
  if (!std::isfinite(mode_time_constant) || mode_time_constant <= 0.0 ||
      !std::isfinite(cv_acceleration_variance) || cv_acceleration_variance <= 0.0 ||
      !std::isfinite(ca_jerk_variance) || ca_jerk_variance <= 0.0)
    throw std::invalid_argument("IMM tuning values must be finite and positive");
  imm_mode_time_constant_ = mode_time_constant;
  cv_acceleration_variance_ = cv_acceleration_variance;
  ca_jerk_variance_ = ca_jerk_variance;
}

bool Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取最近的候选装甲板参与关联。
  for (int i = 0; i < std::min(3, armor_num_); i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    const auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                             std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    const auto position_error = (armor.xyz_in_world - xyza.head(3)).norm();
    const auto error = max_spatial_error_ > 0 ? position_error : angle_error;

    if (error < min_angle_error) {
      id = xyza_i_list[i].second;
      min_angle_error = error;
    }
  }

  if (max_spatial_error_ > 0 && min_angle_error > max_spatial_error_) return false;

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  update_ypda(armor, id);
  return true;
}

void Target::update_ypda(const Armor & armor, int id)
{
  update_model(cv_, armor, id);
  update_model(ca_, armor, id);
  const double cv_log_weight = std::log(std::max(model_probability_[0], 1e-12)) +
                               cv_.data["imm_log_likelihood"];
  const double ca_log_weight = std::log(std::max(model_probability_[1], 1e-12)) +
                               ca_.data["imm_log_likelihood"];
  const double max_log_weight = std::max(cv_log_weight, ca_log_weight);
  model_probability_[0] = std::exp(std::max(-700.0, cv_log_weight - max_log_weight));
  model_probability_[1] = std::exp(std::max(-700.0, ca_log_weight - max_log_weight));
  fuse_models();
  if (max_angular_speed_ > 0)
    ekf_.x[7] = std::clamp(ekf_.x[7], -max_angular_speed_, max_angular_speed_);
  apply_fixed_geometry();
}

void Target::update_model(tools::ExtendedKalmanFilter & filter, const Armor & armor, int id)
{
  const Eigen::MatrixXd H = h_jacobian(filter.x, id);
  const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  const Eigen::VectorXd R_diagonal{
    {4e-3, 4e-3, std::log(std::abs(delta_angle) + 1) + 1,
     std::log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  const Eigen::MatrixXd R = R_diagonal.asDiagonal();
  const Eigen::VectorXd z{{armor.ypd_in_world[0], armor.ypd_in_world[1],
                            armor.ypd_in_world[2], armor.ypr_in_world[0]}};
  auto h = [&](const Eigen::VectorXd & state) -> Eigen::Vector4d {
    const Eigen::Vector3d xyz = h_armor_xyz(state, id);
    const Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
    const auto angle = tools::limit_rad(state[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };
  auto subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    Eigen::VectorXd residual = a - b;
    residual[0] = tools::limit_rad(residual[0]);
    residual[1] = tools::limit_rad(residual[1]);
    residual[3] = tools::limit_rad(residual[3]);
    return residual;
  };
  const Eigen::VectorXd residual = subtract(z, h(filter.x));
  const Eigen::MatrixXd innovation = H * filter.P * H.transpose() + R;
  Eigen::LLT<Eigen::MatrixXd> llt(innovation);
  if (llt.info() == Eigen::Success && residual.allFinite()) {
    const Eigen::VectorXd whitened = llt.matrixL().solve(residual);
    const double log_det = 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    const double log_likelihood = -0.5 *
      (4.0 * std::log(2.0 * CV_PI) + log_det + whitened.squaredNorm());
    filter.data["imm_log_likelihood"] = std::isfinite(log_likelihood) ? log_likelihood : -700.0;
  } else {
    filter.data["imm_log_likelihood"] = -700.0;
  }
  filter.update(z, H, R, h, subtract);
  if (max_angular_speed_ > 0)
    filter.x[7] = std::clamp(filter.x[7], -max_angular_speed_, max_angular_speed_);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  if (!ekf_.x.allFinite() || !ekf_.P.allFinite()) {
    static bool logged_nonfinite_target = false;
    if (!logged_nonfinite_target) {
      logged_nonfinite_target = true;
      tools::logger()->error(
        "[Target] EKF state non-finite: x0={} x2={} x4={} x6={} r={} l={} h={}", ekf_.x[0],
        ekf_.x[2], ekf_.x[4], ekf_.x[6], ekf_.x[8], ekf_.x[9], ekf_.x[10]);
    }
    return true;
  }

  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0, 0, 0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0, 0, 0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh, 0, 0},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0, 0, 0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
