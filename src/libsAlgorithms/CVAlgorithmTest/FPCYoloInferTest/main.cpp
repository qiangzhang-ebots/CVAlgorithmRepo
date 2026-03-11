#include <chrono>
#include <iostream>

#include "FPCYoloTRTInfer/FPCYoloTRTInfer.h"

int main() {
  std::string engine_path =
      "/root/perception/workspace/TRTProject/data/best.engine";
  std::string val_dir =
      "/root/perception/workspace/TRTProject/data/20260301_2026-02-25_15_11_40_043.png";

  // std::string engine_path =
  //     "/home/ebots/Desktop/zhq/FPCDetection/runs/pose/train/weights/"
  //     "best.engine";
  // std::string val_dir =
  //     "/home/ebots/Desktop/zhq/FPCDetection/datasets/images/val/"
  //     "2026-02-25_15_11_12_399.png";

  bool ret = false;
  FPCYoloTRTInfer infer;
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

  // 统计推理时间
  auto start = std::chrono::high_resolution_clock::now();
  int T = 10;
  for (int i = 0; i < T; i++) {
    ret = infer.Predict(input_image);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "Inference time: " << elapsed / T << " ms" << std::endl;

  if (!ret) {
    std::cerr << "Failed to run inference on image " << val_dir << std::endl;
    return -1;
  }

  auto objects = infer.GetObjects();

  for (auto& obj : objects) {
    // cv::rectangle(image, obj.bounding_box, cv::Scalar(0, 255, 0), 2);
    size_t kp_size = obj.keypoints.size();
    for (size_t i = 0; i < kp_size; ++i) {
      cv::circle(input_image, obj.keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
      if (kp_size > 1) {
        cv::line(input_image, obj.keypoints[i],
                 obj.keypoints[(i + 1) % kp_size], cv::Scalar(255, 0, 0), 2);
      }
    }
  }
  // 等比例缩放
  int max_width = 800;
  int max_height = 600;
  double scale = std::min(max_width / (double)input_image.cols,
                          max_height / (double)input_image.rows);
  cv::Mat resized_image;
  cv::resize(input_image, resized_image, cv::Size(), scale, scale);
  cv::imshow("Result", resized_image);
  cv::waitKey(0);

  return 0;
}
