#include "trajectory.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

namespace tools
{
constexpr double g = 9.7833;

namespace
{
struct State
{
  double x, z, vx, vz;
};

State derivative(const State & s, double k)
{
  const double speed = std::hypot(s.vx, s.vz);
  return {s.vx, s.vz, -k * speed * s.vx, -g - k * speed * s.vz};
}

State add(const State & a, const State & b, double scale)
{
  return {a.x + scale * b.x, a.z + scale * b.z, a.vx + scale * b.vx,
          a.vz + scale * b.vz};
}

struct Crossing
{
  bool valid = false;
  double z = 0;
  double time = 0;
};

Crossing at_distance(double speed, double distance, double angle, double k)
{
  State s{0, 0, speed * std::cos(angle), speed * std::sin(angle)};
  double time = 0;
  // Fixed 2 ms RK4 step keeps the interpolation error well below the size of an armor plate.
  constexpr double dt = 0.002;
  for (int step = 0; step < 1500 && s.x < distance; ++step) {
    if (!std::isfinite(s.x) || !std::isfinite(s.z) || s.vx <= 0) return {};
    const State k1 = derivative(s, k);
    const State k2 = derivative(add(s, k1, dt / 2), k);
    const State k3 = derivative(add(s, k2, dt / 2), k);
    const State k4 = derivative(add(s, k3, dt), k);
    const State next{
      s.x + dt * (k1.x + 2 * k2.x + 2 * k3.x + k4.x) / 6,
      s.z + dt * (k1.z + 2 * k2.z + 2 * k3.z + k4.z) / 6,
      s.vx + dt * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx) / 6,
      s.vz + dt * (k1.vz + 2 * k2.vz + 2 * k3.vz + k4.vz) / 6};
    if (next.x >= distance) {
      const double ratio = (distance - s.x) / (next.x - s.x);
      return {true, s.z + ratio * (next.z - s.z), time + ratio * dt};
    }
    s = next;
    time += dt;
  }
  return {};
}
}  // namespace

Trajectory::Trajectory(const double v0, const double d, const double h, double drag_k)
{
  unsolvable = true;
  pitch = fly_time = std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(v0) || !std::isfinite(d) || !std::isfinite(h) ||
      !std::isfinite(drag_k) || v0 <= 0 || d <= 0 || drag_k < 0) return;
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    unsolvable = true;
    return;
  }

  unsolvable = false;
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;

  if (drag_k == 0.0) return;

  // The no-drag low arc is a lower bound for the drag-corrected low arc.
  double lower = pitch;
  auto lower_hit = at_distance(v0, d, lower, drag_k);
  if (!lower_hit.valid) {
    unsolvable = true;
    return;
  }
  double upper = std::min(lower + 0.1, 1.35);
  auto upper_hit = at_distance(v0, d, upper, drag_k);
  while (upper_hit.valid && upper_hit.z < h && upper < 1.35) {
    upper = std::min(upper + 0.1, 1.35);
    upper_hit = at_distance(v0, d, upper, drag_k);
  }
  if (!upper_hit.valid || lower_hit.z > h || upper_hit.z < h) {
    unsolvable = true;
    return;
  }
  for (int iter = 0; iter < 23; ++iter) {
    const double middle = (lower + upper) / 2;
    const auto hit = at_distance(v0, d, middle, drag_k);
    if (!hit.valid) {
      unsolvable = true;
      return;
    }
    if (hit.z < h)
      lower = middle;
    else
      upper = middle;
  }
  pitch = (lower + upper) / 2;
  const auto final_hit = at_distance(v0, d, pitch, drag_k);
  if (!final_hit.valid) {
    unsolvable = true;
    return;
  }
  fly_time = final_hit.time;
}

}  // namespace tools
