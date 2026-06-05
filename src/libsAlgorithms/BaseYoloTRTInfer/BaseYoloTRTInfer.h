#ifndef BASEYOLOTRTINFER_H
#define BASEYOLOTRTINFER_H

#include <opencv2/opencv.hpp>

#include "BaseTRTInfer/BaseTRT.h"
#include "BaseYoloTRTGlobal.h"

struct Params {
  double resize_ratio = 0;
  double dw = 0;
  double dh = 0;
  int height = 0;
  int width = 0;
};

class BASEYOLOINFER_EXPORT BaseYoloTRTInfer : public BaseTRT {
 public:
  BaseYoloTRTInfer();
  ~BaseYoloTRTInfer();

  bool Predict(const cv::Mat& inputImage);

  /*
   * Scale coordinates from the letterboxed image back to the original image
   * size
   */
  cv::Point2f ScaleCoords(const cv::Point2f& point);

 protected:
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
  Params params_;
};

#endif