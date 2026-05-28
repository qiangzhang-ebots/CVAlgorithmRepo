#include "HRNetInfer.h"

#include <limits>

namespace {

cv::Size2f FixAspectRatio(const cv::Size2f& boxScale, float aspectRatio) {
  cv::Size2f adjustedScale = boxScale;
  if (adjustedScale.width <= 0.0f || adjustedScale.height <= 0.0f) {
    return adjustedScale;
  }

  if (adjustedScale.width > adjustedScale.height * aspectRatio) {
    adjustedScale.height = adjustedScale.width / aspectRatio;
  } else {
    adjustedScale.width = adjustedScale.height * aspectRatio;
  }

  return adjustedScale;
}

cv::Matx23f GetUdpWarpMatrix(
  const cv::Point2f& center,
  const cv::Size2f& scale,
  const cv::Size& outputSize) {
  const float outputWidth = static_cast<float>(outputSize.width);
  const float outputHeight = static_cast<float>(outputSize.height);
  const float scaleX = (outputWidth - 1.0f) / scale.width;
  const float scaleY = (outputHeight - 1.0f) / scale.height;

  return cv::Matx23f(
    scaleX,
    0.0f,
    (outputWidth - 1.0f) * 0.5f - center.x * scaleX,
    0.0f,
    scaleY,
    (outputHeight - 1.0f) * 0.5f - center.y * scaleY);
}

BboxTransform GetBboxTransform(const cv::Rect& box, float aspectRatio) {
  BboxTransform bboxTransform;
  bboxTransform.center = cv::Point2f(
    static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f,
    static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f);
  bboxTransform.scale = FixAspectRatio(
    cv::Size2f(static_cast<float>(box.width), static_cast<float>(box.height)),
    aspectRatio);
  return bboxTransform;
}

}  // namespace


/**
 * @brief 对heatmap进行高斯模糊（对应mmpose的gaussian_blur）
 */
cv::Mat gaussian_blur(const cv::Mat& heatmap, int kernel_size) {
  CV_Assert(heatmap.type() == CV_32F);
  CV_Assert(kernel_size > 0 && kernel_size % 2 == 1);

  const int border = (kernel_size - 1) / 2;
  const double origin_max = static_cast<double>(cv::norm(
    heatmap, cv::NORM_INF));

  cv::Mat padded = cv::Mat::zeros(
    heatmap.rows + 2 * border, heatmap.cols + 2 * border, CV_32F);
  heatmap.copyTo(
    padded(cv::Rect(border, border, heatmap.cols, heatmap.rows)));

  cv::Mat blurred;
  cv::GaussianBlur(
    padded, blurred, cv::Size(kernel_size, kernel_size), 0, 0);

  cv::Mat result =
    blurred(cv::Rect(border, border, heatmap.cols, heatmap.rows)).clone();
  const double blurred_max = static_cast<double>(cv::norm(
    result, cv::NORM_INF));
  if (blurred_max > 0.0) {
    result *= static_cast<float>(origin_max / blurred_max);
  }

  return result;
}

/**
 * @brief pad heatmap，mode='edge' 对应复制边缘像素
 */
cv::Mat pad_edge(const cv::Mat& heatmap, int pad_size) {
    cv::Mat result;
    cv::copyMakeBorder(heatmap, result, pad_size, pad_size, pad_size, pad_size, 
                       cv::BORDER_REPLICATE);
    return result;
}

/**
 * @brief UDPHeatmap解码，严格遵循mmpose的refine_keypoints_dark_udp逻辑
 * @param heatmap 模型输出的heatmap（单通道，float类型，shape: H x W）
 * @param inputImg 原始输入图像
 * @param blur_kernel_size 高斯模糊核大小，默认11
 * @return 关键点在原始图像上的坐标
 */
cv::Point2f HRNetInfer::DecodeKeypointsDarkUDP(const cv::Mat& heatmap, const cv::Size2f& bboxScale, int blur_kernel_size)
{
    CV_Assert(heatmap.type() == CV_32F);
    
    int H = heatmap.rows;
    int W = heatmap.cols;
    
    // ========== 步骤1：在heatmap上找到最大值点的整数坐标 ==========
    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(heatmap, nullptr, &max_val, nullptr, &max_loc);
    
    int mx_int = max_loc.x;
    int my_int = max_loc.y;
    
    float mx = static_cast<float>(mx_int);
    float my = static_cast<float>(my_int);
    
    // ========== 步骤2：DARK亚像素精修（严格遵循mmpose逻辑） ==========
    
    // 2.1 高斯模糊
    cv::Mat heatmaps_blur = gaussian_blur(heatmap, blur_kernel_size);
    
    // 2.2 clip到[1e-3, 50]，然后取对数
    cv::Mat heatmaps_clipped;
    cv::max(heatmaps_blur, 1e-3, heatmaps_clipped);
    cv::min(heatmaps_clipped, 50.0, heatmaps_clipped);
    
    cv::Mat log_heatmap;
    cv::log(heatmaps_clipped, log_heatmap);
    
    // 2.3 pad: ((0,0),(1,1),(1,1)), mode='edge'
    cv::Mat heatmaps_pad = pad_edge(log_heatmap, 1);
    
    // 在pad后的heatmap中，原坐标(x,y)对应(x+1, y+1)
    int padded_x = mx_int + 1;
    int padded_y = my_int + 1;

    // 获取3x3邻域各点的值；边界由edge padding处理
    float i_      = heatmaps_pad.at<float>(padded_y,     padded_x    );  // 中心
    float ix1     = heatmaps_pad.at<float>(padded_y,     padded_x + 1);  // 右
    float ix1_    = heatmaps_pad.at<float>(padded_y,     padded_x - 1);  // 左
    float iy1     = heatmaps_pad.at<float>(padded_y + 1, padded_x    );  // 下
    float iy1_    = heatmaps_pad.at<float>(padded_y - 1, padded_x    );  // 上
    float ix1y1   = heatmaps_pad.at<float>(padded_y + 1, padded_x + 1);  // 右下
    float ix1_y1_ = heatmaps_pad.at<float>(padded_y - 1, padded_x - 1);  // 左上

    // ========== 严格按照mmpose公式计算导数 ==========

    // 一阶导数（梯度）
    float dx = 0.5f * (ix1 - ix1_);
    float dy = 0.5f * (iy1 - iy1_);

    // 二阶导数（海森矩阵元素）
    float dxx = ix1 - 2 * i_ + ix1_;
    float dyy = iy1 - 2 * i_ + iy1_;

    // 混合偏导
    float dxy = 0.5f * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_);

    // ========== 构建海森矩阵并求逆 ==========
    // H = [[dxx, dxy], [dxy, dyy]]
    // H⁻¹ = 1/(dxx*dyy - dxy*dxy) * [[dyy, -dxy], [-dxy, dxx]]

    float det = dxx * dyy - dxy * dxy;

    // 对应 mmpose: hessian + eps * I
    const float eps = std::numeric_limits<float>::epsilon();
    det += eps * (dxx + dyy) + eps * eps;

    if (std::abs(det) > 1e-10f) {
      float inv_det = 1.0f / det;
      float offset_x = inv_det * (dyy * dx - dxy * dy);
      float offset_y = inv_det * (-dxy * dx + dxx * dy);

      mx -= offset_x;
      my -= offset_y;
    }
    
    // ========== 步骤3：UDP无偏坐标变换，映射到原始图像坐标 ==========
    float input_w = bboxScale.width;
    float input_h = bboxScale.height;
    
    // UDP方式：除以 (W-1, H-1)，而不是 (W, H)
    float x = mx / (static_cast<float>(W) - 1.0f) * input_w;
    float y = my / (static_cast<float>(H) - 1.0f) * input_h;
    
    return cv::Point2f(x, y);
}

KeypointObjectDescriptor HRNetInfer::Predict(const cv::Mat& inputImage, const cv::Rect& box)
{
  KeypointObjectDescriptor obj; 
  try {
    cv::Mat image;
    if (inputImage.channels() == 1) {
      cv::cvtColor(inputImage, image, cv::COLOR_GRAY2BGR);
    } else {
      image = inputImage;
    }
    const int width = input_binding_.dims.d[3];
    const int height = input_binding_.dims.d[2];
    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    const BboxTransform bboxTransform = GetBboxTransform(box, aspectRatio);

    if (bboxTransform.scale.width <= 0.0f || bboxTransform.scale.height <= 0.0f) {
      return obj;
    }

    bool ret = false;
    ret = Preprocess(image, bboxTransform);
    if (!ret) {
      return obj;
    }
    ret = Infer();
    if (!ret) {
      return obj;
    }
    return Postprocess(bboxTransform);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return obj;
  }

  return obj;
}

KeypointObjectDescriptor HRNetInfer::Postprocess(const BboxTransform& bboxTransform)
{
    // std::vector<cv::Mat> heatmaps_;
    // heatmaps_.clear();
    KeypointObjectDescriptor obj; 

    const int batch = output_binding_.dims.d[0];
    const int num_channels = output_binding_.dims.d[1];
    const int heatmapH = output_binding_.dims.d[2];
    const int heatmapW = output_binding_.dims.d[3];
    const int heatmapSize = heatmapH * heatmapW;

    const float* output_ptr = static_cast<const float*>(host_buffer_);
    if (output_ptr == nullptr) {
        std::cerr << "HRNet output buffer is null" << std::endl;
        return obj;
    }

    // HRNet output is typically NCHW. Here we convert the first batch into a
    // vector of per-channel heatmaps.
    const int batch_offset = 0;
    for (int c = 0; c < num_channels; ++c) {
        const float* channel_ptr =
            output_ptr + ((batch_offset * num_channels + c) * heatmapSize);
        cv::Mat heatmap(heatmapH, heatmapW, CV_32F,
                        const_cast<float*>(channel_ptr));

        double heatmap_max = 0.0;
        cv::minMaxLoc(heatmap, nullptr, &heatmap_max);

        // heatmaps_.push_back(heatmap.clone());
        cv::Point2f pos = DecodeKeypointsDarkUDP(heatmap, bboxTransform.scale);
        pos.x += bboxTransform.center.x - bboxTransform.scale.width * 0.5f;
        pos.y += bboxTransform.center.y - bboxTransform.scale.height * 0.5f;
        obj.keypoints.push_back(pos);
        obj.confidences.push_back(static_cast<float>(heatmap_max));
    }

    return obj; 
}

bool HRNetInfer::Preprocess(const cv::Mat& input_image, const BboxTransform& bboxTransform) {
  if (input_image.empty() || bboxTransform.scale.width <= 0.0f ||
      bboxTransform.scale.height <= 0.0f) {
    return false;
  }

  auto width = input_binding_.dims.d[3];
  auto height = input_binding_.dims.d[2];
  cv::Size size(width, height);

  const cv::Matx23f warpMatrix = GetUdpWarpMatrix(
    bboxTransform.center, bboxTransform.scale, size);
  cv::Mat resized_image;
  cv::warpAffine(
    input_image,
    resized_image,
    cv::Mat(warpMatrix),
    size,
    cv::INTER_LINEAR,
    cv::BORDER_CONSTANT,
    cv::Scalar(0, 0, 0));


  float* input_ptr = static_cast<float*>(host_input_buffer_);
  const int image_area = width * height;
  double m[3] = {123.675, 116.28, 103.53};
  double s[3] = {58.395, 57.12, 57.375};
  for (int row = 0; row < height; ++row) {
    const cv::Vec3b* row_ptr = resized_image.ptr<cv::Vec3b>(row);
    for (int col = 0; col < width; ++col) {
      const cv::Vec3b& pixel = row_ptr[col];
      const int index = row * width + col;
      input_ptr[index] = (pixel[2]-m[0])/s[0];
      input_ptr[image_area + index] = (pixel[1]-m[1])/s[1];
      input_ptr[2 * image_area + index] = (pixel[0]-m[2])/s[2];
    }
  }

  cudaError_t err;
  err = cudaMemcpyAsync(device_buffers_[0], host_input_buffer_,
                        input_binding_.size, cudaMemcpyHostToDevice, stream_);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data to device for input tensor "
              << input_binding_.name << std::endl;
    return false;
  }
  return true;
}