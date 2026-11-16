#include <chrono>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/yolos/yolov5.hpp"

// Offline detector probe: never connects to a camera, simulator, or controller.
int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "Usage: yolov5_image_probe CONFIG IMAGE_OR_VIDEO [max_frames=1] [stride=1] "
                 "[frame_directory]\n";
    return 2;
  }
  auto_aim::YOLOV5 detector(argv[1], false);
  cv::Mat still = cv::imread(argv[2]);
  cv::VideoCapture video;
  if (still.empty() && !video.open(argv[2])) return 2;
  const int limit = argc > 3 ? std::stoi(argv[3]) : 1;
  const int stride = argc > 4 ? std::stoi(argv[4]) : 1;
  for (int i = 0; i < limit; ++i) {
    cv::Mat frame = still;
    if (still.empty()) {
      video.set(cv::CAP_PROP_POS_FRAMES, i * stride);
      if (!video.read(frame)) break;
    }
    if (argc > 5) {
      std::filesystem::create_directories(argv[5]);
      cv::imwrite(
        (std::filesystem::path(argv[5]) / (std::to_string(i * stride) + ".png")).string(), frame);
    }
    const auto start = std::chrono::steady_clock::now();
    auto armors = detector.detect(frame, i);
    const auto ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    nlohmann::json result = {
      {"frame", i * stride}, {"ms", ms}, {"detections", nlohmann::json::array()}};
    for (const auto & armor : armors) {
      result["detections"].push_back(
        {{"name", auto_aim::ARMOR_NAMES.at(armor.name)},
         {"color", auto_aim::COLORS.at(armor.color)},
         {"confidence", armor.confidence},
         {"number_confidence", armor.number_confidence},
         {"number_margin", armor.number_margin},
         {"center", {armor.center.x, armor.center.y}}});
    }
    std::cout << result.dump() << '\n';
  }
}
