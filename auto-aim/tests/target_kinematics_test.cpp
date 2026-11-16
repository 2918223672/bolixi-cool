#include <cmath>
#include <stdexcept>

#include "tasks/auto_aim/target.hpp"

namespace
{
void expect_near(double actual, double expected)
{
  if (std::abs(actual - expected) > 1e-9)
    throw std::runtime_error("armor rotation prediction has the wrong sign");
}

void check_spin(double omega, double center_x)
{
  constexpr double radius = 0.25;
  constexpr double dt = 0.1;
  auto_aim::Target target(center_x, omega, radius, 0.0);
  const auto before = target.armor_xyza_list().front();
  expect_near(before.x(), center_x - radius);
  expect_near(before.y(), 0.0);
  const auto velocity = target.armor_velocity(0);
  expect_near(velocity.x(), 0.0);
  expect_near(velocity.y(), -radius * omega);
  const double line_of_sight_rate = before.x() * velocity.y() /
                                    (before.x() * before.x());
  expect_near(line_of_sight_rate, -radius * omega / (center_x - radius));

  target.predict(dt);
  const auto after = target.armor_xyza_list().front();
  expect_near(after.x(), center_x - radius * std::cos(omega * dt));
  expect_near(after.y(), -radius * std::sin(omega * dt));
  expect_near(after[3], omega * dt);
  const auto future_velocity = target.armor_velocity(0);
  expect_near(future_velocity.x(), radius * omega * std::sin(omega * dt));
  expect_near(future_velocity.y(), -radius * omega * std::cos(omega * dt));
  if (after.y() * omega >= 0.0)
    throw std::runtime_error("armor tangential velocity must oppose positive yaw at the front");
}
}  // namespace

int main()
{
  for (double distance : {1.5, 2.0, 4.0, 6.0}) {
    check_spin(4.0, distance);
    check_spin(-4.0, distance);
  }
}
