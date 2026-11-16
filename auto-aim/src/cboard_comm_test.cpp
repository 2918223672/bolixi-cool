#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "io/cboard.hpp"
#include "tools/exiter.hpp"

using namespace std::chrono_literals;

namespace
{
double parse_angle(const std::string & value)
{
  size_t parsed = 0;
  const double angle = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(angle))
    throw std::invalid_argument("yaw and pitch must be finite radians");
  return angle;
}

void usage(const char * program)
{
  std::cout << "Usage: " << program
            << " [config.yaml] [--control --yaw=RADIANS --pitch=RADIANS]\n";
}
}  // namespace

int main(int argc, char * argv[])
{
  std::string config_path = "configs/standard3.yaml";
  io::Command command{false, false, 0, 0};
  bool has_config = false;
  bool has_yaw = false;
  bool has_pitch = false;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        usage(argv[0]);
        return 0;
      }
      if (arg == "--control") {
        command.control = true;
      } else if (arg.rfind("--yaw=", 0) == 0) {
        command.yaw = parse_angle(arg.substr(6));
        has_yaw = true;
      } else if (arg.rfind("--pitch=", 0) == 0) {
        command.pitch = parse_angle(arg.substr(8));
        has_pitch = true;
      } else if (!arg.empty() && arg[0] != '-' && !has_config) {
        config_path = arg;
        has_config = true;
      } else {
        throw std::invalid_argument("Unknown or repeated argument: " + arg);
      }
    }
    if (command.control != (has_yaw && has_pitch) || has_yaw != has_pitch)
      throw std::invalid_argument("--control requires both --yaw and --pitch");

    tools::Exiter exiter;
    io::CBoard cboard(config_path);
    std::cout << "CBoard ready; configured bullet speed: " << cboard.bullet_speed
              << " m/s. Press Ctrl+C to stop.\n";

    while (!exiter.exit()) {
      try {
        const auto q = cboard.imu_at(std::chrono::steady_clock::now() - 1ms);
        std::cout << "IMU xyzw: " << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
                  << " | TX control=" << command.control << " shoot=0 yaw=" << command.yaw
                  << " pitch=" << command.pitch << std::endl;
        cboard.send(command);
      } catch (const std::exception & e) {
        cboard.send({false, false, 0, 0});
        std::cerr << "CBoard receive error: " << e.what() << std::endl;
      }
      std::this_thread::sleep_for(100ms);
    }

    cboard.send({false, false, 0, 0});
    return 0;
  } catch (const std::exception & e) {
    std::cerr << e.what() << std::endl;
    usage(argv[0]);
    return 1;
  }
}
