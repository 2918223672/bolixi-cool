#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <fmt/core.h>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "io/sim_client.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path | configs/standard3.yaml | 实车 YAML 配置文件路径}"
  "{channel | default | 仿真器 IPC 通道}"
  "{display | true | 显示检测画面}"
  "{max-frames | 0 | 处理指定帧数后退出，0 表示持续运行}"
  "{stop-on-kill | false | 击毁目标后停火、等待弹丸结算并退出}"
  "{expected-health | 0 | 要求测试开始时的满血值，0 表示不校验}"
  "{diagnostics | false | 记录击发时的 EKF、瞄准点和云台误差}"
  "{trace | | 逐帧诊断 JSONL 路径，留空关闭}";

namespace
{
std::string detection_labels(const std::list<auto_aim::Armor> & armors)
{
  std::string labels;
  for (const auto & armor : armors) {
    if (!labels.empty()) labels += ',';
    labels += fmt::format(
      "{}(class={},conf={:.2f})", auto_aim::ARMOR_NAMES.at(armor.name), armor.class_id,
      armor.confidence);
  }
  return labels.empty() ? "none" : labels;
}

std::string visible_plate_ids(const nlohmann::json & frame_metadata)
{
  std::string ids;
  for (const auto & armor : frame_metadata.at("camera").at("armors")) {
    if (!ids.empty()) ids += ',';
    ids += std::to_string(armor.value("id", -1));
  }
  return ids.empty() ? "none" : ids;
}

void validate_context(const YAML::Node & yaml, const io::SimFrame & frame)
{
  const auto & camera = frame.metadata.at("camera");
  if (
    camera.at("width").get<int>() != frame.bgr.cols ||
    camera.at("height").get<int>() != frame.bgr.rows)
    throw std::runtime_error("Simulator frame dimensions do not match camera metadata");

  const auto expected_team = yaml["enemy_color"].as<std::string>();
  const auto actual_team = frame.metadata.at("appearance").at("team").get<std::string>();
  if (actual_team != expected_team)
    throw std::runtime_error(
      "Simulator target team is " + actual_team + " but enemy_color is " + expected_team);
}

void draw_debug(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors, const io::Command & command,
  const nlohmann::json & state)
{
  for (const auto & armor : armors) {
    for (std::size_t i = 0; i < armor.points.size(); ++i)
      cv::line(
        image, armor.points[i], armor.points[(i + 1) % armor.points.size()], {0, 255, 0}, 2);
    cv::putText(
      image, auto_aim::ARMOR_NAMES.at(armor.name), armor.center, cv::FONT_HERSHEY_SIMPLEX, 0.7,
      {0, 255, 255}, 2);
  }
  const auto & stats = state.at("stats");
  const std::string status = cv::format(
    "detect=%zu control=%d fire=%d hit=%d/%d %.1f%%", armors.size(), command.control,
    command.shoot, stats.value("hit", 0), stats.value("resolved", 0),
    stats.value("hit_rate", 0.0) * 100.0);
  cv::putText(image, status, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2);
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);
  const bool display = cli.get<bool>("display");
  const int max_frames = cli.get<int>("max-frames");
  const bool stop_on_kill = cli.get<bool>("stop-on-kill");
  const int expected_health = cli.get<int>("expected-health");
  const bool diagnostics = cli.get<bool>("diagnostics");
  const auto yaml = YAML::LoadFile(config_path);
  const auto channel = cli.get<std::string>("channel");

  try {
    tools::Exiter exiter;
    std::ofstream trace;
    const auto trace_path = cli.get<std::string>("trace");
    if (!trace_path.empty()) {
      trace.open(trace_path);
      if (!trace) throw std::runtime_error("Cannot open trace file: " + trace_path);
    }
    io::SimClient client(channel);
    auto_aim::YOLO detector(config_path, false);
    auto_aim::Solver solver(config_path);
    std::unique_ptr<auto_aim::Tracker> tracker;
    std::unique_ptr<auto_aim::Aimer> aimer;
    std::unique_ptr<auto_aim::Shooter> shooter;
    auto next_report = std::chrono::steady_clock::now();
    int processed_frames = 0;
    int detected_frames = 0;
    int observable_frames = 0;
    int observable_detected_frames = 0;
    int tracking_frames = 0;
    int control_frames = 0;
    bool test_started = false;
    bool killed = false;
    double start_time = 0;
    double first_detection_time = -1;
    double first_tracking_time = -1;
    double first_control_time = -1;
    double first_hit_time = -1;
    double kill_time = -1;
    int initial_fired = 0;
    int initial_resolved = 0;
    int initial_hit = 0;
    int initial_body = 0;
    int initial_ground = 0;
    int initial_expired = 0;
    int initial_damage = 0;
    int initial_kills = 0;
    io::Command previous_command{false, false, 0, 0};
    double command_yaw_step = 0;
    double gimbal_yaw_error = 0;
    double gimbal_pitch_error = 0;

    for (int frame_count = 0; !exiter.exit(); ++frame_count) {
      if (max_frames > 0 && frame_count >= max_frames) break;
      // Godot can temporarily stop drawing while its window is minimized or while the desktop
      // compositor restores it.  Give only the first capture a longer startup allowance; later
      // frames retain the normal watchdog so a broken live camera does not go unnoticed.
      auto frame = client.next_frame(frame_count == 0 ? 60s : 10s);
      if (frame.context_changed) {
        client.send({false, false, 0, 0});
        validate_context(yaml, frame);
        const auto & camera = frame.metadata.at("camera");
        solver.set_camera_calibration(
          camera.at("K").get<std::vector<double>>(),
          camera.at("distortion").get<std::vector<double>>(),
          camera.at("R_camera2gimbal").get<std::vector<double>>(),
          camera.at("t_camera2gimbal").get<std::vector<double>>());
        solver.set_R_gimbal2imubody(Eigen::Matrix3d::Identity());
        tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
        aimer = std::make_unique<auto_aim::Aimer>(config_path);
        shooter = std::make_unique<auto_aim::Shooter>(config_path);
        tools::logger()->info(
          "Simulator context synchronized: channel={}, session={}, camera={}x{}", channel,
          frame.metadata.at("session_id").get<std::string>(), frame.bgr.cols, frame.bgr.rows);
        if (test_started)
          throw std::runtime_error("Simulator session changed during the test; restart this run");
      }

      if (!test_started) {
        client.refresh_state();
        const auto & state = client.state();
        const auto & health = state.at("health");
        const auto & stats = state.at("stats");
        const int maximum_health = health.at("maximum").get<int>();
        const int current_health = health.at("current").get<int>();
        if (expected_health > 0 && maximum_health != expected_health)
          throw std::runtime_error(
            "Simulator maximum health is " + std::to_string(maximum_health) + ", expected " +
            std::to_string(expected_health));
        if (stop_on_kill && current_health != maximum_health)
          throw std::runtime_error(
            "Target is not at full health; reset/restart the simulator before a timed test");
        start_time = frame.sim_time;
        initial_fired = stats.value("fired", 0);
        initial_resolved = stats.value("resolved", 0);
        initial_hit = stats.value("hit", 0);
        initial_body = stats.value("body", 0);
        initial_ground = stats.value("ground", 0);
        initial_expired = stats.value("expired", 0);
        initial_damage = stats.value("damage", 0);
        initial_kills = stats.value("kills", 0);
        test_started = true;
        tools::logger()->info(
          "Timed test started: sim={:.3f}s health={}/{} fired={} hit={}", start_time,
          current_health, maximum_health, initial_fired, initial_hit);
      }

      solver.set_R_gimbal2world(frame.gimbal_to_world);
      const auto processing_start = std::chrono::steady_clock::now();
      auto armors = detector.detect(frame.bgr, frame_count);
      auto targets = tracker->track(armors, frame.timestamp);
      client.refresh_state();
      const double processing_delay =
        std::clamp(client.state_sim_time() - frame.sim_time, 0.0, 0.5);
      // Predict from the estimated current simulator time, not the older image timestamp.
      const auto aim_timestamp =
        frame.timestamp + std::chrono::microseconds(static_cast<int64_t>(processing_delay * 1e6));
      const double shoot_state_time = client.state_sim_time();
      const auto & current_gimbal = client.state().at("gimbal");
      const Eigen::Vector3d gimbal_pos{
        current_gimbal.at("yaw").get<double>(), current_gimbal.at("pitch").get<double>(), 0.0};
      auto command = aimer->aim(
        targets, aim_timestamp, frame.bullet_speed, false);
      command_yaw_step = std::abs(tools::limit_rad(previous_command.yaw - command.yaw));
      gimbal_yaw_error = std::abs(tools::limit_rad(gimbal_pos[0] - previous_command.yaw));
      gimbal_pitch_error = std::abs(tools::limit_rad(gimbal_pos[1] - previous_command.pitch));
      command.shoot = shooter->shoot(command, *aimer, targets, gimbal_pos);
      if (diagnostics && command.shoot && !targets.empty()) {
        const auto state = targets.front().ekf_x();
        const auto aim = aimer->debug_aim_point.xyza;
        tools::logger()->info(
          "SHOT_INPUT sim={:.3f}s recognized={} expected={} plate_id={} visible_plate_ids={} "
          "center=({:.3f},{:.3f},{:.3f}) velocity=({:.3f},{:.3f},"
          "{:.3f}) armor_yaw={:.3f} angular_speed={:.3f} aim=({:.3f},{:.3f},{:.3f}) "
          "command=({:.3f},{:.3f}) gimbal=({:.3f},{:.3f}) delay={:.1f}ms",
          frame.sim_time, detection_labels(armors),
          frame.metadata.at("appearance").value("robot", "unknown"), targets.front().last_id,
          visible_plate_ids(frame.metadata), state[0], state[2], state[4], state[1], state[3],
          state[5], state[6], state[7], aim[0], aim[1], aim[2], command.yaw, command.pitch,
          gimbal_pos[0], gimbal_pos[1], processing_delay * 1e3);
      }
      client.send(command);
      if (trace.is_open()) {
        // Truth is recorded for offline comparison only; it is never passed to the controller.
        const auto & snapshot = client.state();
        nlohmann::json row{
          {"frame", frame_count}, {"frame_time", frame.sim_time},
          {"session", frame.metadata.at("session_id")},
          {"processing_delay", processing_delay},
          {"processing_wall_ms", std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - processing_start).count()},
          {"capture_timing", frame.metadata.value("capture_timing", nlohmann::json::object())},
          {"frame_gimbal", frame.metadata.at("gimbal")},
          {"truth_camera_armors", frame.metadata.at("camera").at("armors")},
          {"detected", armors.size()}, {"recognized", detection_labels(armors)},
          {"tracker_state", tracker->state()},
          {"command", {{"control", command.control}, {"shoot", command.shoot},
                       {"yaw", command.yaw}, {"pitch", command.pitch}}},
          {"fire_command", {{"valid", aimer->fire_command.control},
                            {"yaw", aimer->fire_command.yaw},
                            {"pitch", aimer->fire_command.pitch}}},
          {"fire_window", aimer->debug_aim_point.valid},
          {"state_time", snapshot.at("sim_time")},
          {"shoot_state_time", shoot_state_time},
          {"shoot_gimbal", {{"yaw", gimbal_pos[0]}, {"pitch", gimbal_pos[1]}}},
          {"gimbal", snapshot.at("gimbal")}, {"truth", snapshot.at("truth")},
          {"stats", snapshot.at("stats")}, {"events", snapshot.at("events")}};
        row["detections"] = nlohmann::json::array();
        for (const auto & armor : armors) {
          nlohmann::json points = nlohmann::json::array();
          for (const auto & point : armor.points) points.push_back({point.x, point.y});
          row["detections"].push_back({
            {"name", auto_aim::ARMOR_NAMES.at(armor.name)},
            {"type", static_cast<int>(armor.type)},
            {"points", points},
            {"xyz", {armor.xyz_in_world.x(), armor.xyz_in_world.y(), armor.xyz_in_world.z()}},
            {"yaw", armor.ypr_in_world[0]}, {"yaw_raw", armor.yaw_raw}});
        }
        if (!targets.empty()) {
          const auto & target = targets.front();
          const auto x = target.ekf_x();
          row["ekf"] = std::vector<double>(x.data(), x.data() + x.size());
          row["ekf_data"] = target.ekf().data;
          row["plate_id"] = target.last_id;
          row["jumped"] = target.jumped;
          row["estimated_armors"] = nlohmann::json::array();
          for (const auto & armor : target.armor_xyza_list())
            row["estimated_armors"].push_back({armor[0], armor[1], armor[2], armor[3]});
          if (!armors.empty() && armors.front().name == target.name &&
              armors.front().type == target.armor_type) {
            const auto & observed = armors.front();
            row["observation"] = {
              {"xyz", {observed.xyz_in_world.x(), observed.xyz_in_world.y(),
                       observed.xyz_in_world.z()}},
              {"yaw_raw", observed.yaw_raw}, {"yaw", observed.ypr_in_world[0]}};
          }
        }
        if (command.control) {
          const auto & point = aimer->debug_aim_point.xyza;
          row["aim"] = std::vector<double>(point.data(), point.data() + point.size());
          const auto & prediction = aimer->debug_prediction;
          if (prediction.valid) {
            const auto seconds = [](std::chrono::steady_clock::time_point t) {
              return std::chrono::duration<double>(t.time_since_epoch()).count();
            };
            row["prediction"] = {
              {"reference_time", seconds(prediction.reference_timestamp)},
              {"arrival_time", seconds(prediction.prediction_timestamp)},
              {"fly_time", prediction.fly_time}, {"iterations", prediction.iterations},
              {"converged", prediction.converged}};
          }
        }
        trace << row.dump() << '\n';
      }
      previous_command = command;
      processed_frames++;
      detected_frames += !armors.empty();
      const bool target_observable = !frame.metadata.at("camera").at("armors").empty();
      observable_frames += target_observable;
      observable_detected_frames += target_observable && !armors.empty();
      tracking_frames += !targets.empty();
      control_frames += command.control;
      if (first_detection_time < 0 && !armors.empty()) first_detection_time = frame.sim_time;
      if (first_tracking_time < 0 && !targets.empty()) first_tracking_time = frame.sim_time;
      if (first_control_time < 0 && command.control) first_control_time = frame.sim_time;

      const auto & current_state = client.state();
      const auto & current_stats = current_state.at("stats");
      if (first_hit_time < 0 && current_stats.value("hit", 0) > initial_hit)
        first_hit_time = current_stats.value("first_hit_time", frame.sim_time);
      if (
        !current_state.at("health").value("alive", true) &&
        current_stats.value("kills", 0) > initial_kills) {
        killed = true;
        kill_time = current_state.at("health").value("death_time", frame.sim_time);
      }

      if (std::chrono::steady_clock::now() >= next_report) {
        const auto & stats = client.state().at("stats");
        Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
        double center_z = 0;
        double vertical_speed = 0;
        double nis = 0;
        bool nis_failed = false;
        int armor_id = -1;
        if (!targets.empty()) {
          velocity = Eigen::Vector3d(
            targets.front().ekf_x()[1], targets.front().ekf_x()[3],
            targets.front().ekf_x()[7]);
          center_z = targets.front().ekf_x()[4];
          vertical_speed = targets.front().ekf_x()[5];
          nis = targets.front().ekf().last_nis;
          nis_failed = targets.front().ekf().data.at("nis_fail") > 0.5;
          armor_id = targets.front().last_id;
        }
        if (diagnostics)
          tools::logger()->info(
            "sim={:.2f}s detected={} tracking={} control={} fire={} hit={}/{} ({:.1f}%) "
            "delay={:.1f}ms velocity=({:.2f},{:.2f}) center_z={:.3f} vz={:.3f} "
            "angular_speed={:.2f} recognized={} expected={} plate_id={} visible_plate_ids={} "
            "nis={:.2f} rejected={} yaw_step={:.2f}deg gimbal_error=({:.2f},{:.2f})deg",
            frame.sim_time, armors.size(), targets.size(), command.control, command.shoot,
            stats.value("hit", 0), stats.value("resolved", 0),
            stats.value("hit_rate", 0.0) * 100.0, processing_delay * 1e3, velocity[0], velocity[1],
            center_z, vertical_speed, velocity[2], detection_labels(armors),
            frame.metadata.at("appearance").value("robot", "unknown"), armor_id,
            visible_plate_ids(frame.metadata), nis, nis_failed,
            command_yaw_step * 180.0 / CV_PI, gimbal_yaw_error * 180.0 / CV_PI,
            gimbal_pitch_error * 180.0 / CV_PI);
        else
          tools::logger()->info(
            "sim={:.2f}s detected={} tracking={} control={} fire={} hit={}/{} ({:.1f}%) "
            "delay={:.1f}ms",
            frame.sim_time, armors.size(), targets.size(), command.control, command.shoot,
            stats.value("hit", 0), stats.value("resolved", 0),
            stats.value("hit_rate", 0.0) * 100.0, processing_delay * 1e3);
        next_report = std::chrono::steady_clock::now() + 1s;
      }

      if (display) {
        auto drawing = frame.bgr.clone();
        draw_debug(drawing, armors, command, client.state());
        cv::imshow("cool-sentry rm_aim_sim", drawing);
        const auto key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
      }
      if (stop_on_kill && killed) break;
    }
    client.send({false, false, 0, 0});

    // Let already-fired projectiles resolve so fired/resolved and hit rate describe the same run.
    if (stop_on_kill && killed) {
      const auto deadline = std::chrono::steady_clock::now() + 4s;
      while (std::chrono::steady_clock::now() < deadline) {
        client.refresh_state();
        if (client.state().at("stats").value("in_flight", 0) == 0) break;
        std::this_thread::sleep_for(10ms);
      }
    }

    client.refresh_state();
    const auto & final_state = client.state();
    const auto & final_stats = final_state.at("stats");
    const int fired = final_stats.value("fired", 0) - initial_fired;
    const int resolved = final_stats.value("resolved", 0) - initial_resolved;
    const int hit = final_stats.value("hit", 0) - initial_hit;
    const int body = final_stats.value("body", 0) - initial_body;
    const int ground = final_stats.value("ground", 0) - initial_ground;
    const int expired = final_stats.value("expired", 0) - initial_expired;
    const int damage = final_stats.value("damage", 0) - initial_damage;
    const int kills = final_stats.value("kills", 0) - initial_kills;
    const double detection_rate =
      observable_frames ? 100.0 * observable_detected_frames / observable_frames : 0.0;
    const double raw_detection_rate =
      processed_frames ? 100.0 * detected_frames / processed_frames : 0.0;
    const double tracking_rate =
      processed_frames ? 100.0 * tracking_frames / processed_frames : 0.0;
    const double control_rate =
      processed_frames ? 100.0 * control_frames / processed_frames : 0.0;
    const double hit_rate = resolved ? 100.0 * hit / resolved : 0.0;
    const double time_to_kill = killed ? kill_time - start_time : -1.0;
    const double detection_to_kill =
      killed && first_detection_time >= 0 ? kill_time - first_detection_time : -1.0;
    const double first_hit_to_kill =
      killed && first_hit_time >= 0 ? kill_time - first_hit_time : -1.0;
    tools::logger()->info(
      "SIM_SUMMARY frames={} detected_when_visible={}/{} ({:.1f}%) raw_detected={}/{} ({:.1f}%) "
      "tracking={}/{} ({:.1f}%) control={}/{} ({:.1f}%)",
      processed_frames, observable_detected_frames, observable_frames, detection_rate,
      detected_frames, processed_frames, raw_detection_rate, tracking_frames, processed_frames,
      tracking_rate, control_frames, processed_frames, control_rate);
    tools::logger()->info(
      "SIM_RESULT health={} killed={} time_to_kill={:.3f}s detection_to_kill={:.3f}s "
      "first_hit_to_kill={:.3f}s first_detection={:.3f}s first_tracking={:.3f}s "
      "first_control={:.3f}s fired={} resolved={} hit={} hit_rate={:.1f}% damage={} kills={} "
      "body={} ground={} expired={} detection_rate={:.1f}% raw_detection_rate={:.1f}% "
      "tracking_rate={:.1f}% control_rate={:.1f}%",
      final_state.at("health").value("maximum", 0), killed, time_to_kill, detection_to_kill,
      first_hit_to_kill, first_detection_time < 0 ? -1.0 : first_detection_time - start_time,
      first_tracking_time < 0 ? -1.0 : first_tracking_time - start_time,
      first_control_time < 0 ? -1.0 : first_control_time - start_time, fired, resolved, hit,
      hit_rate, damage, kills, body, ground, expired, detection_rate, raw_detection_rate,
      tracking_rate, control_rate);
    if (display) cv::destroyAllWindows();
    return 0;
  } catch (const std::exception & error) {
    tools::logger()->error("auto_aim_sim stopped: {}", error.what());
    return 1;
  }
}
