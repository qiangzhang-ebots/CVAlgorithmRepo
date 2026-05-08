#include <chrono>
#include <iostream>

#include "HRNetInfer/HRNetInfer.h"

int main() {
  std::string engine_path =
      "/root/perception/workspace/model_data/HRN.engine";
  std::string val_dir =
      "/root/perception/workspace/ebots_perception_ros2/test_data/image_detector/Xiaomi.png";

  bool ret = false;
  HRNetInfer infer;
  infer.SetGPUId(1);
  ret = infer.LoadModel(engine_path);
  if (!ret) {
    std::cerr << "Failed to load model from " << engine_path << std::endl;
    return -1;
  }

  cv::Mat input_image = cv::imread(val_dir);
  if (input_image.empty()) {
    std::cerr << "Failed to read image from " << val_dir << std::endl;
    return -1;
  }

//   cv::Point2d p1(730, 310), p2(1016, 470);
//   cv::Rect roi(cv::Point(static_cast<int>(p1.x), static_cast<int>(p1.y)),
//                cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y)));
  cv::Rect roi(744-10, 466-10, 351+20, 184+20);
  // cv::Rect roi(966-10, 788-10, 263+20, 178+20);
  cv::Mat patch = input_image(roi).clone();
  std::cout << "patch size: " << patch.cols << " x " << patch.rows << std::endl;
  
  // 统计推理时间
  auto start = std::chrono::high_resolution_clock::now();
  KeypointObjectDescriptor obj;
  int T = 100;
  for (int i = 0; i < T; i++) {
    obj = infer.Predict(input_image, roi);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Inference time: " << elapsed / T << " ms" << std::endl;


    size_t kp_size = obj.keypoints.size();
    for (size_t i = 0; i < kp_size; ++i) {
      cv::circle(input_image, obj.keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
      if (kp_size > 1) {
        cv::line(input_image, obj.keypoints[i],
                 obj.keypoints[(i + 1) % kp_size], cv::Scalar(255, 0, 0), 2);
      }
    }


  return 0;
}
