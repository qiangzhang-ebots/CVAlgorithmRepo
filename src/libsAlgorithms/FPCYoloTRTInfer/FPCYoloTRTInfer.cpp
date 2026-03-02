
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