#include "extended_kalman_filter.hpp"

#include <numeric>

#include "tools/logger.hpp"
//卡尔曼滤波 (EKF)理解它是如何根据目标过去的位置，预测出子弹飞行到目标所需的时间后，目标会在哪里的。
namespace tools
{
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  if ((!x.allFinite() || !P.allFinite())) {
    static bool logged_nonfinite_predict = false;
    if (!logged_nonfinite_predict) {
      logged_nonfinite_predict = true;
      tools::logger()->error(
        "[EKF] non-finite after predict: x_finite={} P_finite={}", x.allFinite(),
        P.allFinite());
    }
  }
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  if (!x.allFinite() || !P.allFinite() || !z.allFinite() || !H.allFinite() || !R.allFinite()) {
    static bool logged_nonfinite_input = false;
    if (!logged_nonfinite_input) {
      logged_nonfinite_input = true;
      tools::logger()->error(
        "[EKF] non-finite input: x={} P={} z={} H={} R={}", x.allFinite(), P.allFinite(),
        z.allFinite(), H.allFinite(), R.allFinite());
    }
    return x;
  }

  Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd innovation = H * P * H.transpose() + R;
  if (!innovation.allFinite()) {
    static bool logged_nonfinite_innovation = false;
    if (!logged_nonfinite_innovation) {
      logged_nonfinite_innovation = true;
      tools::logger()->error("[EKF] non-finite innovation covariance; x={} P={} H={} R={}",
                             x.allFinite(), P.allFinite(), H.allFinite(), R.allFinite());
    }
    return x;
  }
  Eigen::MatrixXd K = P * H.transpose() * innovation.inverse();
  if (!K.allFinite()) {
    static bool logged_nonfinite_gain = false;
    if (!logged_nonfinite_gain) {
      logged_nonfinite_gain = true;
      tools::logger()->error(
        "[EKF] non-finite Kalman gain; S_finite={}", innovation.allFinite());
    }
    return x;
  }

  // Stable Compution of the Posterior Covariance
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x, K * z_subtract(z, h(x)));
  if (!x.allFinite() || !P.allFinite()) {
    static bool logged_nonfinite_update = false;
    if (!logged_nonfinite_update) {
      logged_nonfinite_update = true;
      tools::logger()->error(
        "[EKF] non-finite after update: x={} P={} z={} K={}", x.allFinite(), P.allFinite(),
        z.allFinite(), K.allFinite());
    }
  }

  /// 卡方检验
  Eigen::VectorXd residual = z_subtract(z, h(x));
  // 新增检验
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  double nis = residual.transpose() * S.inverse() * residual;
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  // 卡方检验阈值（自由度=4，取置信水平95%）
  constexpr double nis_threshold = 0.711;
  constexpr double nees_threshold = 0.711;

  if (nis > nis_threshold) nis_count_++, data["nis_fail"] = 1;
  if (nees > nees_threshold) nees_count_++, data["nees_fail"] = 1;
  total_count_++;
  last_nis = nis;

  recent_nis_failures.push_back(nis > nis_threshold ? 1 : 0);

  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  data["residual_yaw"] = residual[0];
  data["residual_pitch"] = residual[1];
  data["residual_distance"] = residual[2];
  data["residual_angle"] = residual[3];
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace tools
