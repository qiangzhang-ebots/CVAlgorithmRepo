
#include "FPCYoloTRTInfer.h"

template <typename T>
T clamp(T value, T min, T max) {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

FPCYoloTRTInfer::FPCYoloTRTInfer() {
  // Constructor implementation
}

FPCYoloTRTInfer::~FPCYoloTRTInfer() {}

void FPCYoloTRTInfer::PostprocessOneObject(const float* row_ptr) {
  YoloKeypointObjectDescriptor obj;
  auto bboxes_ptr = row_ptr;
  auto scores_ptr = row_ptr + 4;   // Assuming first 4 are bbox,
  auto kps_ptr = row_ptr + 4 + 1;  // Assuming 1 class score, then

  float score = *scores_ptr;
  if (score > 0.5)  // Confidence threshold
  {
    float x1 = *bboxes_ptr++;
    float y1 = *bboxes_ptr++;
    float x2 = *bboxes_ptr++;
    float y2 = *bboxes_ptr++;

    cv::Point2f p1 = ScaleCoords(cv::Point2f(x1, y1));
    cv::Point2f p2 = ScaleCoords(cv::Point2f(x2, y2));
    x1 = p1.x;
    y1 = p1.y;
    x2 = p2.x;
    y2 = p2.y;

    float label = *kps_ptr++;
    obj.label = static_cast<int>(label);
    obj.box_confidence = score;

    // float x0 = clamp<float>((x-0.5f*w)*width_ratio, 0, width);
    // float y0 = clamp<float>((y-0.5f*h)*height_ratio, 0, height);
    // float x1 = clamp<float>((x+0.5f*w)*width_ratio, 0, width);
    // float y1 = clamp<float>((y+0.5f*h)*height_ratio, 0, height);
    cv::Rect bbox;
    bbox.x = x1;
    bbox.y = y1;
    bbox.width = x2 - x1;
    bbox.height = y2 - y1;
    obj.bounding_box = bbox;

    for (int k = 0; k < 4; k++)  // Assuming 17 keypoints
    {
      // float kpx = (*(kps_ptr + 3*k) - dw)*width_ratio;
      // float kpy = (*(kps_ptr + 3*k + 1) - dh)*height_ratio;
      // float kps = *(kps_ptr + 3*k + 2); // Keypoint confidence
      float kpx = *(kps_ptr + 3 * k);
      float kpy = *(kps_ptr + 3 * k + 1);
      float kps = *(kps_ptr + 3 * k + 2);

      cv::Point2f kp = ScaleCoords(cv::Point2f(kpx, kpy));
      kpx = kp.x;
      kpy = kp.y;

      kpx = clamp<float>(kpx, 0, params_.width);
      kpy = clamp<float>(kpy, 0, params_.height);

      if (kps > 0.5)  // Keypoint confidence threshold
      {
        obj.keypoints.emplace_back(kpx, kpy);
      }

      obj.confidences.push_back(kps);
    }

    fpc_zif_objs_.push_back(obj);
  }
}

std::vector<YoloKeypointObjectDescriptor> FPCYoloTRTInfer::GetObjects() {
  
  std::vector<YoloKeypointObjectDescriptor> valid_objs;

  for (int i = 0; i < 4; i++)
  {
    YoloKeypointObjectDescriptor best_obj; 
    for (const auto& obj : fpc_zif_objs_) {
      if (obj.box_confidence > 0.5 && obj.label == i) {
        if (obj.box_confidence > best_obj.box_confidence) 
        {
          best_obj = obj;
        }
      }
    }
    valid_objs.push_back(best_obj);
  }
  m_valid_objs_ = valid_objs;
  return valid_objs;
}

void FPCYoloTRTInfer::Postprocess() {
  fpc_zif_objs_.clear();
  BaseYoloTRTInfer::Postprocess();
}

double CalculateOverlap(const YoloKeypointObjectDescriptor& FPC, const YoloKeypointObjectDescriptor& ZIF) {
  if (FPC.keypoints.size() < 3 || ZIF.keypoints.size() < 3) {
    return 0.0;  // Not enough keypoints to form a polygon
  }

    // std::vector<cv::Point2f> fpc_poly, zif_poly;
    // for (const auto& kp : FPC.keypoints) fpc_poly.push_back(kp);
    // for (const auto& kp : ZIF.keypoints) zif_poly.push_back(kp);

  std::vector<cv::Point2f> intersection_poly;
  double intersection_area = cv::intersectConvexConvex(FPC.keypoints, ZIF.keypoints, intersection_poly);
  double fpc_area = cv::contourArea(FPC.keypoints);

  if (fpc_area == 0) return 0.0;
  return intersection_area / fpc_area;
}

double CalculateArea(const YoloKeypointObjectDescriptor& FPC) {
  if (FPC.keypoints.size() < 3) {
    return 0.0;
  }

  return cv::contourArea(FPC.keypoints);
}

std::pair<double, double> FPCYoloTRTInfer::CalOverLap() {
  if (fpc_zif_objs_.empty()) {
    return std::make_pair(0.0, 0.0);
  }
  if (m_valid_objs_.empty()) {
    GetObjects();
  }

  auto fpc1 = m_valid_objs_[0];
  auto zif1 = m_valid_objs_[1];
  auto fpc2 = m_valid_objs_[2];
  auto zif2 = m_valid_objs_[3];

  double overlap1 = CalculateOverlap(fpc1, zif1);
  double overlap2 = CalculateOverlap(fpc2, zif2);

  return std::make_pair(overlap1, overlap2);
}

std::pair<double, double> FPCYoloTRTInfer::CalFpcArea() {
  if (fpc_zif_objs_.empty()) {
    return std::make_pair(0.0, 0.0);
  }
  if (m_valid_objs_.empty()) {
    GetObjects();
  }

  auto fpc1 = m_valid_objs_[0];
  auto fpc2 = m_valid_objs_[2];

  return std::make_pair(CalculateArea(fpc1), CalculateArea(fpc2));
}
