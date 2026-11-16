#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/yolos/yolov5.hpp"

int main()
{
  auto_aim::YOLOV5 detector("configs/standard3.yaml", false);
  cv::Mat output = cv::Mat::zeros(2, 22, CV_32F);
  for (int row = 0; row < output.rows; ++row) {
    output.at<float>(row, 0) = 100;
    output.at<float>(row, 1) = 100;
    output.at<float>(row, 6) = 140;
    output.at<float>(row, 7) = 100;
    output.at<float>(row, 4) = 140;
    output.at<float>(row, 5) = 120;
    output.at<float>(row, 2) = 100;
    output.at<float>(row, 3) = 120;
    output.at<float>(row, 10) = 1;  // Red.
  }
  output.at<float>(0, 8) = 4;  // Higher-scoring outpost must not suppress armor four.
  output.at<float>(0, 19) = 1;
  output.at<float>(1, 8) = 2.5;
  output.at<float>(1, 17) = 1;

  const cv::Mat image = cv::Mat::zeros(640, 640, CV_8UC3);
  const auto armors = detector.postprocess(1.0, output, image, 0);
  if (
    armors.size() != 1 || armors.front().name != auto_aim::ArmorName::four ||
    armors.front().color != auto_aim::Color::red) {
    std::cerr << "Valid armor was suppressed by an invalid candidate\n";
    return 1;
  }
  // High objectness must not be mistaken for certainty about the digit.
  cv::Mat ambiguous = output.row(1).clone();
  ambiguous.at<float>(0, 16) = 1;  // Equal scores for 3 and 4.
  if (!detector.postprocess(1.0, ambiguous, image, 0).empty()) {
    std::cerr << "Ambiguous digit escaped the classification margin gate\n";
    return 1;
  }
  cv::Mat uncertain = output.row(1).clone();
  uncertain.colRange(13, 22).setTo(-5);
  uncertain.at<float>(0, 17) = -2;
  if (!detector.postprocess(1.0, uncertain, image, 0).empty()) {
    std::cerr << "Low digit score escaped the classification confidence gate\n";
    return 1;
  }
  cv::Mat weak = output.row(1).clone();
  weak.at<float>(0, 8) = 0.85f;  // Objectness ~0.70, below production threshold.
  if (!detector.postprocess(1.0, weak, image, 0).empty()) {
    std::cerr << "A weak proposal was promoted without successful re-detection\n";
    return 1;
  }
  cv::Mat corrupt = output.row(1).clone();
  corrupt.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
  if (!detector.postprocess(1.0, corrupt, image, 0).empty()) return 1;

  for (int number : {0, 1, 3, 4, 5}) {
    cv::Mat row = output.row(1).clone();
    row.colRange(13, 22).setTo(-5);
    row.at<float>(0, 13 + number) = 5;
    const auto found = detector.postprocess(1.0, row, image, 0);
    const auto expected = number == 0 ? auto_aim::sentry : auto_aim::ArmorName(number - 1);
    if (
      found.size() != 1 || found.front().name != expected || found.front().class_id != number ||
      found.front().type != (number == 1 ? auto_aim::big : auto_aim::small)) {
      std::cerr << "Hero/sentry/infantry class mapping changed\n";
      return 1;
    }
  }

  // Verify scale and ROI offset are applied once, including the NMS rectangle.
  const auto path =
    std::filesystem::temp_directory_path() /
    ("yolov5-roi-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
     ".yaml");
  auto config = YAML::LoadFile("configs/standard3.yaml");
  config["use_roi"] = true;
  config["use_traditional"] = false;
  {
    std::ofstream file(path);
    file << config;
  }
  auto_aim::YOLOV5 roi_detector(path.string(), false);
  std::filesystem::remove(path);
  cv::Mat roi_output = output.row(1).clone();
  const cv::Mat large_image = cv::Mat::zeros(1080, 1440, CV_8UC3);
  const auto offset_armors = roi_detector.postprocess(0.5, roi_output, large_image, 0);
  if (
    offset_armors.size() != 1 ||
    cv::norm(offset_armors.front().center - cv::Point2f(660, 270)) > 0.01 ||
    offset_armors.front().box.x != 620 || offset_armors.front().box.y != 250) {
    std::cerr << "ROI coordinates were transformed incorrectly\n";
    return 1;
  }
  std::cout << "YOLOv5 NMS, digit gates, class mapping, invalid input and ROI tests passed\n";
  return 0;
}
