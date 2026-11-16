#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

private:
  Solver & solver_;
  Color enemy_color_;
  double max_frame_gap_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  bool fixed_normal_geometry_ = false;
  double max_spatial_error_ = 0;
  double max_angular_speed_ = 0;
  double imm_mode_time_constant_ = 0.35;
  double cv_acceleration_variance_ = 20.0;
  double ca_jerk_variance_ = 300.0;
  double max_armor_height_error_ = 0;
  double max_coast_image_error_px_ = 0;
  int max_spin_coast_frames_ = 0;
  int invalid_coast_count_ = 0;
  ArmorName forced_armor_name_ = ArmorName::not_armor;
  double normal_radius_ = 0;
  double normal_radius_difference_ = 0;
  double normal_height_difference_ = 0;
  double normal_center_z_ = 0;
  std::string state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
  bool valid_observation(const Armor & armor) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
