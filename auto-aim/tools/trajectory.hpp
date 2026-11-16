#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  // drag_k=0 时保持原无阻力解析解；drag_k>0 使用三维二次阻力在弹道平面内积分。
  // drag_k = rho * Cd * A / (2 * mass)，单位：1/m。
  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  Trajectory(const double v0, const double d, const double h, double drag_k = 0.0);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
