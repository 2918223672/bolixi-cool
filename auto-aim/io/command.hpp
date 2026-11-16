#ifndef IO__COMMAND_HPP
#define IO__COMMAND_HPP

namespace io
{
struct Command
{
  bool control;
  bool shoot;
  double yaw;
  double pitch;
  double yaw_vel = 0.0;
  double pitch_vel = 0.0;
};

}  // namespace io

#endif  // IO__COMMAND_HPP
