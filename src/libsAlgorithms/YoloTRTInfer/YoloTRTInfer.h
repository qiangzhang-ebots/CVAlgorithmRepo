#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "BaseTRTInfer/BaseTRT.h"
#include "YoloTRTGlobal.h"

// 任务类型枚举
enum class YoloTaskType {
    DETECT,     // 目标检测: x1,y1,x2,y2
    OBB,        // 旋转目标检测: 4个点表示旋转矩形
    POSE,       // 关键点检测: N个关键点
    SEGMENT     // 分割: N个点表示轮廓
};

// 统一检测结果数据结构
struct YoloObject {
    int label = -1;                              // 类别ID
    std::string class_name;                      // 类别名称(可以为空)
    float confidence = 0.0f;                     // 置信度
    std::vector<cv::Point2f> points;             // 点集（根据任务类型有不同含义）
    std::vector<float> point_confidences;        // 各点置信度（用于关键点检测）
    cv::Rect2f bbox;                             // 目标矩形框（最小外接矩形框）
};

class YOLOINFER_EXPORT YoloTRTInfer : public BaseTRT {
 public:
  YoloTRTInfer(YoloTaskType task_type = YoloTaskType::DETECT);
  ~YoloTRTInfer();

  std::vector<YoloObject> Predict(const cv::Mat& inputImage);

  /*
   * 获取检测结果
   */
  std::vector<YoloObject> GetObjects() const;

  /*
   * 设置任务类型
   */
  void SetTaskType(YoloTaskType task_type);

  /*
   * 获取当前任务类型
   */
  YoloTaskType GetTaskType() const;
  // Scale coordinates from the letterboxed image back to the original image
  cv::Point2f ScaleCoords(const cv::Point2f& point);

protected:
  virtual void Postprocess();
private:
  // 前向声明内部实现类
  struct Impl;
  std::unique_ptr<Impl> impl_;
  
  // 内部方法
  bool Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& new_shape);
  bool Preprocess(const cv::Mat& input_image);
  //void Postprocess();
  void ScalePoints(std::vector<cv::Point2f>& points);
  void CreateStrategy(YoloTaskType task_type);

};


/* how to use
// 创建目标检测器
YoloTRTInfer detector(YoloTaskType::DETECT);
detector.LoadModel("detect.engine");
auto objects = detector.Predict(image);  // 直接返回结果

// 动态切换为关键点检测
detector.SetTaskType(YoloTaskType::POSE);
detector.LoadModel("pose.engine");
auto pose_objects = detector.Predict(image);  // 直接返回结果
*/