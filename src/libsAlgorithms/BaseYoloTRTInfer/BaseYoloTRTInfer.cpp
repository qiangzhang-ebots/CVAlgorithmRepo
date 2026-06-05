#include "BaseYoloTRTInfer.h"

BaseYoloTRTInfer::BaseYoloTRTInfer() {}

BaseYoloTRTInfer::~BaseYoloTRTInfer() {}

bool BaseYoloTRTInfer::Predict(const cv::Mat& input_image) {
  try {
    cv::Mat image;
    if (input_image.channels() == 1) {
      cv::cvtColor(input_image, image, cv::COLOR_GRAY2BGR);
    } else {
      image = input_image;
    }
    bool ret = false;
    ret = Preprocess(image);
    if (!ret) {
      return false;
    }
    ret = Infer();
    if (!ret) {
      return false;
    }
    Postprocess();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return false;
  }

  return true;
}

bool BaseYoloTRTInfer::Preprocess(const cv::Mat& input_image) {
  cv::Mat letterboxed;

  auto width = input_binding_.dims.d[3];
  auto height = input_binding_.dims.d[2];
  cv::Size size(width, height);

  Letterbox(input_image, letterboxed,
            size);  // Implement letterbox resizing to fit model input size

  float* input_ptr = static_cast<float*>(host_input_buffer_);
  const int image_area = width * height;
  const float scale = 1.0f / 255.0f;
  for (int row = 0; row < height; ++row) {
    const cv::Vec3b* row_ptr = letterboxed.ptr<cv::Vec3b>(row);
    for (int col = 0; col < width; ++col) {
      const cv::Vec3b& pixel = row_ptr[col];
      const int index = row * width + col;
      input_ptr[index] = pixel[2] * scale;
      input_ptr[image_area + index] = pixel[1] * scale;
      input_ptr[2 * image_area + index] = pixel[0] * scale;
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

bool BaseYoloTRTInfer::Letterbox(const cv::Mat& image, cv::Mat& output,
                              const cv::Size& size) {
  float inp_h = size.height;
  float inp_w = size.width;
  float height = image.rows;
  float width = image.cols;

  float r = std::min(inp_w / width, inp_h / height);
  float wr = r;
  float hr = r;

  int padw = std::round(width * wr);
  int padh = std::round(height * hr);

  cv::Mat tmp;
  if ((int)width != padw || (int)height != padh) {
    cv::resize(image, tmp, cv::Size(padw, padh));
  } else {
    tmp = image.clone();
  }

  float dw = inp_w - padw;
  float dh = inp_h - padh;

  dw /= 2;
  dh /= 2;

  int top = int(std::round(dh - 0.1));
  int bottom = int(std::round(dh + 0.1));
  int left = int(std::round(dw - 0.1));
  int right = int(std::round(dw + 0.1));

    cv::copyMakeBorder(tmp, output, top, bottom, left, right, cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114));

  params_.resize_ratio = r;
  params_.dw = dw;
  params_.dh = dh;
  params_.height = height;
  params_.width = width;
  return true;
}

void BaseYoloTRTInfer::Postprocess() {
  auto num_channels = output_binding_.dims.d[1];
  auto num_anchors = output_binding_.dims.d[2];
  auto dw = params_.dw;
  auto dh = params_.dh;
  auto width = params_.width;
  auto height = params_.height;
  auto width_ratio = params_.resize_ratio;
  auto height_ratio = params_.resize_ratio;

  cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F,
                           static_cast<float*>(host_buffer_));
  // output = output.t(); // Transpose to (num_anchors, num_channels)
  for (int i = 0; i < num_anchors; ++i) {
    float* data = output.ptr<float>(i);
    PostprocessOneObject(data);
  }
  
}

cv::Point2f BaseYoloTRTInfer::ScaleCoords(const cv::Point2f& point) {
  float x = (point.x - params_.dw) / params_.resize_ratio;
  float y = (point.y - params_.dh) / params_.resize_ratio;
  x = std::min(std::max(x, 0.0f), params_.width - 1.0f);
  y = std::min(std::max(y, 0.0f), params_.height - 1.0f);
  return cv::Point2f(x, y);
}