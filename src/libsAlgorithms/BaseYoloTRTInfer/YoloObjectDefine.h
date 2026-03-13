
#ifndef YOLOOBJECTDEFINE_H
#define YOLOOBJECTDEFINE_H

#include <opencv2/opencv.hpp>

struct YoloKeypointObjectDescriptor {
  int label = -1;              // Class label
  cv::Rect bounding_box;  // OpenCV rectangle for the bounding box
  float box_confidence = 0;   // Confidence score for the bounding box

  std::vector<cv::Point2f> keypoints;  // Vector of keypoints (x, y) coordinates
  std::vector<float> confidences;      // Confidence scores for each keypoint
};

#endif