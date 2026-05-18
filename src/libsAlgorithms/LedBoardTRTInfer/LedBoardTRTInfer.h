#pragma once

#include "BaseYoloTRTInfer/BaseYoloTRTInfer.h"
#include "BaseYoloTRTInfer/YoloObjectDefine.h"
#include <unordered_map>
#include <string>

// 类别名称映射表 - 将类别序号映射到类别名称
const std::unordered_map<int, std::string> LED_BOARD_CLASS_NAMES = {
    {0, "LED_Board_A1"},
    {1, "LED_Board_A2"},
    {2, "LED_Board_B1"},
    {3, "LED_Board_B2"}
};

// 类别到关键点数量的映射
const std::unordered_map<int, int> LED_BOARD_CLASS_KEYPOINT_NUM = {
    {0, 2},   // LED_Board_A1 有2个关键点
    {1, 2},   // LED_Board_A2 有2个关键点
    {2, 4},   // LED_Board_B1 有4个关键点
    {3, 4}    // LED_Board_B2 有4个关键点
};

struct LedBoardObject {
  int label = -1;                           // 类别序号
  std::string class_name;                   // 类别名称
  float box_confidence = 0.0f;              // 边界框置信度
  cv::Rect bounding_box;                    // 边界框（最小外接矩）
  std::vector<cv::Point2f> keypoints;       // 关键点坐标
  std::vector<float> keypoint_confidences;  // 各关键点置信度
};


// 天马LED版 PCB 粘贴多个关键点检测
class LedBoardTRTInfer : public BaseYoloTRTInfer {
public:
  LedBoardTRTInfer();
  ~LedBoardTRTInfer();

  // 定义类别到关键点数量的映射
  static const std::unordered_map<int, int> CLASS_KEYPOINT_NUM;
  
  // 获取检测到的所有目标
  std::vector<LedBoardObject> GetObjects();
  
protected:
  virtual void Postprocess() override;
  void PostprocessOneObject(const float* output) override;

private:
  std::vector<LedBoardObject> detected_objects_;
};