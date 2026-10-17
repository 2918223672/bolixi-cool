#include "extended_kalman_filter.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
//卡尔曼滤波 (EKF)理解它是如何根据目标过去的位置，预测出子弹飞行到目标所需的时间后，目标会在哪里的。
namespace tools
{
namespace
{
double chi_square_upper_95(Eigen::Index degrees_of_freedom)
{
  constexpr std::array<double, 7> thresholds{0,        3.841459,  5.991465, 7.814728,
                                             9.487729, 11.070498, 12.591587};
  if (degrees_of_freedom > 0 && degrees_of_freedom < thresholds.size())
    return thresholds[degrees_of_freedom];
  return std::numeric_limits<double>::infinity();
}
}  // namespace

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
  data["nis_threshold"] = 0.0;
  data["nis_fail"] = 0.0;
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
  return x;
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(
    z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  const Eigen::VectorXd x_prior = x;
  const Eigen::MatrixXd P_prior = P;
  const Eigen::VectorXd innovation = z_subtract(z, h(x_prior));
  const Eigen::MatrixXd S = H * P_prior * H.transpose() + R;
  const Eigen::MatrixXd K =
    P_prior * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

  // Stable Compution of the Posterior Covariance
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  P = (I - K * H) * P_prior * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x_prior, K * innovation);

  const double nis = innovation.dot(S.ldlt().solve(innovation));
  const double nis_threshold = chi_square_upper_95(z.rows());
  const bool nis_failed = !std::isfinite(nis) || nis > nis_threshold;
  if (nis_failed) ++nis_count_;
  total_count_++;
  last_nis = nis;
  last_nis_threshold = nis_threshold;

  recent_nis_failures.push_back(nis_failed ? 1 : 0);

  if (recent_nis_failures.size() > window_size) {
    recent_nis_failures.pop_front();
  }

  int recent_failures = std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0);
  double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

  if (innovation.size() > 0) data["residual_yaw"] = innovation[0];
  if (innovation.size() > 1) data["residual_pitch"] = innovation[1];
  if (innovation.size() > 2) data["residual_distance"] = innovation[2];
  if (innovation.size() > 3) data["residual_angle"] = innovation[3];
  data["nis"] = nis;
  data["nis_threshold"] = nis_threshold;
  data["nis_fail"] = nis_failed ? 1.0 : 0.0;
  data["recent_nis_failures"] = recent_rate;

  return x;
}

}  // namespace tools
