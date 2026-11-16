#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  const auto refinement = yaml["yolo_refinement"];
  if (refinement) {
    refine_enabled_ = refinement["enabled"].as<bool>(false);
    max_refine_rois_ = refinement["max_rois"].as<int>(2);
    proposal_confidence_ = refinement["proposal_confidence"].as<double>(0.35);
    min_number_confidence_ = refinement["min_number_confidence"].as<double>(0.3);
    min_number_margin_ = refinement["min_number_margin"].as<double>(0.1);
  }
  auto probability = [](double value) { return std::isfinite(value) && value >= 0 && value <= 1; };
  if (
    !probability(min_confidence_) ||
    (refine_enabled_ &&
     (!probability(proposal_confidence_) || proposal_confidence_ > min_confidence_ ||
      !probability(min_number_confidence_) || !probability(min_number_margin_) ||
      max_refine_rois_ < 0 || max_refine_rois_ > 4)))
    throw std::invalid_argument("Invalid YOLOv5 confidence/refinement settings");
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);
  auto model = core_.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  infer_request_ = compiled_model_.create_infer_request();
}

cv::Mat YOLOV5::infer(const cv::Mat & image, double & scale)
{
  scale = std::min(640.0 / image.rows, 640.0 / image.cols);
  const int h = std::max(1, static_cast<int>(image.rows * scale));
  const int w = std::max(1, static_cast<int>(image.cols * scale));
  cv::Mat input(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::resize(image, input(cv::Rect(0, 0, w, h)), {w, h});
  infer_request_.set_input_tensor(ov::Tensor(ov::element::u8, {1, 640, 640, 3}, input.data));
  infer_request_.infer();
  const auto output = infer_request_.get_output_tensor();
  const auto shape = output.get_shape();
  if (shape.size() != 3 || shape[2] != 22)
    throw std::runtime_error("YOLOv5 requires [1,N,22] armor model output");
  // The request is reused for local crops; retain full-frame predictions independently.
  return cv::Mat(shape[1], shape[2], CV_32F, output.data()).clone();
}

bool YOLOV5::reliable_number(const Armor & armor) const
{
  return armor.number_confidence >= min_number_confidence_ &&
         armor.number_margin >= min_number_margin_;
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) return {};
  const cv::Rect bounds(0, 0, raw_img.cols, raw_img.rows);
  cv::Rect search = bounds;
  if (use_roi_) {
    search = cv::Rect(
               roi_.x, roi_.y, roi_.width == -1 ? raw_img.cols : roi_.width,
               roi_.height == -1 ? raw_img.rows : roi_.height) &
             bounds;
    if (search.empty()) return {};
  }
  double scale;
  const auto output = infer(raw_img(search), scale);
  const double threshold = refine_enabled_ ? proposal_confidence_ : min_confidence_;
  auto candidates = suppress(decode(scale, output, raw_img, search.tl(), threshold), threshold);
  if (refine_enabled_) {
    int attempts = 0;
    for (auto & candidate : candidates) {
      const int extent = std::max(candidate.box.width, candidate.box.height);
      if (
        extent * scale >= 64 && reliable_number(candidate) &&
        candidate.confidence > min_confidence_)
        continue;
      if (attempts >= max_refine_rois_) break;
      // Context follows the detected object; never use simulator position or identity.
      const int side = std::min({std::clamp(extent * 16, 320, 960), search.width, search.height});
      if (side <= 0 || 640.0 / side <= scale * 1.1) continue;
      const int x = std::clamp(
        static_cast<int>(candidate.center.x) - side / 2, search.x, search.x + search.width - side);
      const int y = std::clamp(
        static_cast<int>(candidate.center.y) - side / 2, search.y, search.y + search.height - side);
      const cv::Rect crop(x, y, side, side);
      double local_scale;
      const auto local_output = infer(raw_img(crop), local_scale);
      ++attempts;
      const auto local_candidates =
        decode(local_scale, local_output, raw_img, crop.tl(), min_confidence_);
      const Armor * best = nullptr;
      double best_quality = 0;
      for (const auto & local : local_candidates) {
        if (!reliable_number(local)) continue;
        const double intersection = (candidate.box & local.box).area();
        const double overlap =
          intersection / (candidate.box.area() + local.box.area() - intersection);
        // Reclassification may change number/type, but must refer to the same image object.
        if (
          overlap < 0.25 || cv::norm(candidate.center - local.center) > std::max(8.0, extent * 0.5))
          continue;
        const double quality = local.confidence * local.number_confidence;
        if (quality > best_quality) {
          best = &local;
          best_quality = quality;
        }
      }
      if (best) {
        const bool rescue = candidate.confidence <= min_confidence_ || !reliable_number(candidate);
        const bool same_label_improvement = best->name == candidate.name &&
                                            best->color == candidate.color &&
                                            best->confidence > candidate.confidence;
        const bool stronger_digit = best->number_confidence > candidate.number_confidence &&
                                    best->number_margin > candidate.number_margin;
        if (rescue || same_label_improvement || stronger_digit) candidate = *best;
      }
    }
  }
  return finish(candidates, raw_img, frame_count);
}

std::vector<Armor> YOLOV5::decode(
  double scale, const cv::Mat & output, const cv::Mat & image, const cv::Point2f & offset,
  double threshold) const
{
  if (!std::isfinite(scale) || scale <= 0 || output.type() != CV_32F || output.cols != 22)
    throw std::invalid_argument("Invalid YOLOv5 output or image scale");
  std::vector<Armor> candidates;
  for (int r = 0; r < output.rows; ++r) {
    const auto row = output.ptr<float>(r);
    if (!std::all_of(row, row + 22, [](float v) { return std::isfinite(v); })) continue;
    const double score = sigmoid(row[8]);
    if (score <= threshold) continue;
    const int color = std::max_element(row + 9, row + 13) - (row + 9);
    const int number = std::max_element(row + 13, row + 22) - (row + 13);
    const double digit_score = sigmoid(row[13 + number]);
    double runner_up = 0;
    for (int n = 0; n < 9; ++n)
      if (n != number) runner_up = std::max(runner_up, sigmoid(row[13 + n]));
    std::vector<cv::Point2f> points;
    for (int index : {0, 6, 4, 2})
      points.emplace_back(row[index] / scale + offset.x, row[index + 1] / scale + offset.y);
    float min_x = points[0].x, max_x = min_x, min_y = points[0].y, max_y = min_y;
    for (const auto & point : points) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    if (
      !std::isfinite(min_x + max_x + min_y + max_y) || min_x < -image.cols ||
      max_x > 2.0 * image.cols || min_y < -image.rows || max_y > 2.0 * image.rows ||
      max_x - min_x < 2 || max_y - min_y < 2)
      continue;
    const cv::Rect box(
      static_cast<int>(min_x), static_cast<int>(min_y), static_cast<int>(max_x - min_x),
      static_cast<int>(max_y - min_y));
    Armor candidate(color, number, score, box, points);
    candidate.class_id = number;
    candidate.number_confidence = digit_score;
    candidate.number_margin = digit_score - runner_up;
    if (
      candidate.center.x < 0 || candidate.center.x >= image.cols || candidate.center.y < 0 ||
      candidate.center.y >= image.rows)
      continue;
    // Invalid labels cannot suppress valid overlapping detections during NMS.
    if (
      candidate.name == ArmorName::not_armor || candidate.name == ArmorName::outpost ||
      !check_type(candidate))
      continue;
    candidates.push_back(candidate);
  }
  return candidates;
}

std::vector<Armor> YOLOV5::suppress(const std::vector<Armor> & candidates, double threshold) const
{
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto & armor : candidates) {
    boxes.push_back(armor.box);
    scores.push_back(armor.confidence);
  }
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, threshold, nms_threshold_, indices);
  std::vector<Armor> kept;
  for (int index : indices) kept.push_back(candidates[index]);
  return kept;
}

std::list<Armor> YOLOV5::finish(
  const std::vector<Armor> & candidates, const cv::Mat & image, int frame_count)
{
  std::vector<Armor> accepted;
  for (const auto & armor : candidates)
    if (check_name(armor) && (!refine_enabled_ || reliable_number(armor)))
      accepted.push_back(armor);
  const auto kept = suppress(accepted, min_confidence_);
  std::list<Armor> armors(kept.begin(), kept.end());
  tmp_img_ = image;
  for (auto & armor : armors) {
    if (use_traditional_) detector_.detect(armor, image);
    armor.center_norm = get_center_norm(image, armor.center);
  }
  if (debug_) draw_detections(image, armors, frame_count);
  return armors;
}

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & image, int frame_count)
{
  return finish(
    decode(scale, output, image, use_roi_ ? offset_ : cv::Point2f(), min_confidence_), image,
    frame_count);
}

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor && armor.name != ArmorName::outpost;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
