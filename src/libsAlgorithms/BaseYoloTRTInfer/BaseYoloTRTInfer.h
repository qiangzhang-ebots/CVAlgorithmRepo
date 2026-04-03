#ifndef BASEYOLOTRTINFER_H
#define BASEYOLOTRTINFER_H

#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <opencv2/opencv.hpp>
#include <string>

#include "BaseYoloTRTGlobal.h"

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cout << "TensorRT: " << msg << std::endl;
    }
  }
};

struct Binding {
  std::string name;
  nvinfer1::DataType dtype;
  nvinfer1::Dims dims;
  void* buffer;  // Device buffer
  int size = 0;
};

struct Params {
  double resize_ratio = 0;
  double dw = 0;
  double dh = 0;
  int height = 0;
  int width = 0;
};

class BASEYOLOINFER_EXPORT BaseYoloTRTInfer {
 public:
  BaseYoloTRTInfer();
  ~BaseYoloTRTInfer();

  bool LoadModel(const std::string& modelPath);
  bool Predict(const cv::Mat& inputImage);

  /*
   * Scale coordinates from the letterboxed image back to the original image
   * size
   */
  cv::Point2f ScaleCoords(const cv::Point2f& point);

 protected:
  void MakePipe(bool is_warmup = false);
  bool Infer();

  // yolo use letterbox to resize input image, so we need to calculate the
  // padding and resize ratio for postprocess
  bool Letterbox(const cv::Mat& src, cv::Mat& dst, const cv::Size& new_shape);
  virtual bool Preprocess(const cv::Mat& input_image);
  virtual void Postprocess();

  // for one object, postprocess the output of model. It would be
  // segmentation/detection/pose estimation, etc. every child class of
  // BaseYoloInfer should implement this function to postprocess the output of
  // model for one object.
  virtual void PostprocessOneObject(const float* output) = 0;

 protected:
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  Logger logger_;

  Binding input_binding_;
  Binding output_binding_;

  void* device_buffers_[2] = {nullptr, nullptr};  // Pointers to input and output device buffers
  void* host_input_buffer_ = nullptr;
  void* host_buffer_ = nullptr;        // Pointer to output host buffer
  Params params_;
};

#endif