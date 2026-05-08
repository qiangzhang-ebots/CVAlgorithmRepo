#ifndef HRNETINFER_H
#define HRNETINFER_H


#include "HRNetInferGlobal.h"
#include "BaseTRTInfer/BaseTRT.h"
#include "BaseTRTInfer/KeypointObjectDefine.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct BboxTransform {
    cv::Point2f center;
    cv::Size2f scale;
};

class HRNETINFER_EXPORT HRNetInfer: public BaseTRT
{

public:


    KeypointObjectDescriptor Predict(const cv::Mat& inputImage, const cv::Rect& box);

    /*
    * 采用 mmpose 的 UDP 方式，按 bbox scale 解码 heatmap
    */
    cv::Point2f DecodeKeypointsDarkUDP(const cv::Mat& heatmap, const cv::Size2f& bboxScale, int blur_kernel_size=11);
protected:
    virtual bool Preprocess(const cv::Mat& input_image, const BboxTransform& bboxTransform);
    virtual KeypointObjectDescriptor Postprocess(const BboxTransform& bboxTransform);
};

#endif