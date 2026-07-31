#include "YoloTRTInfer.h"
#include "YoloPostprocessStrategy.h"


// 后处理策略接口
class PostprocessStrategy {
public:
    virtual ~PostprocessStrategy() = default;
    
    virtual void Postprocess(
        const float* output_buffer,
        int num_channels,
        int num_anchors,
        std::vector<YoloObject>& detected_objects) = 0;
};

// 内部实现结构体
struct YoloTRTInfer::Impl {
    // 参数
    double resize_ratio = 0;
    double dw = 0;
    double dh = 0;
    int height = 0;
    int width = 0;
    
    // 任务类型和策略
    YoloTaskType task_type;
    using PostprocessFunc = std::function<void( const float*, int, int, std::vector<YoloObject>&)>;
    PostprocessFunc postprocess_func;
    
    // 类别信息
    std::vector<std::string> class_names;           // 类别名称列表
    std::vector<int> class_kps;                     // 每个类别的关键点数量，用于关键点检测，其他任务可能不需要
    
    // 检测结果
    std::vector<YoloObject> detected_objects;
};

// YoloTRTInfer 实现
YoloTRTInfer::YoloTRTInfer(YoloTaskType task_type) 
    : impl_(std::make_unique<Impl>()) {
    impl_->task_type = task_type;
    CreateStrategy(task_type);
}

YoloTRTInfer::~YoloTRTInfer() = default;

std::vector<YoloObject> YoloTRTInfer::Predict(const cv::Mat& input_image) {
  if (!SetCudaDevice("YoloTRTInfer::Predict")) {
    return impl_->detected_objects;
  }

  impl_->detected_objects.clear();
  
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
      return impl_->detected_objects;
    }
    ret = Infer();
    if (!ret) {
      return impl_->detected_objects;
    }
    Postprocess();
  } catch (const std::exception& e) {
    std::cerr << CV_ALGORITHM_LOG_PREFIX << e.what() << '\n';
    return impl_->detected_objects;
  }

  return impl_->detected_objects;
}

std::vector<YoloObject> YoloTRTInfer::GetObjects() const {
    return impl_->detected_objects;
}

YoloTaskType YoloTRTInfer::GetTaskType() const {
    return impl_->task_type;
}

void YoloTRTInfer::SetTaskType(YoloTaskType task_type) {
    impl_->task_type = task_type;
    CreateStrategy(task_type);
}

bool YoloTRTInfer::Preprocess(const cv::Mat& input_image) {
  cv::Mat letterboxed;

  auto width = input_binding_.dims.d[3];
  auto height = input_binding_.dims.d[2];
  cv::Size size(width, height);

  Letterbox(input_image, letterboxed, size);

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
    std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to copy data to device for input tensor "
              << input_binding_.name << ": " << cudaGetErrorString(err)
              << std::endl;
    return false;
  }
  return true;
}

bool YoloTRTInfer::Letterbox(const cv::Mat& image, cv::Mat& output,
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

  impl_->resize_ratio = r;
  impl_->dw = dw;
  impl_->dh = dh;
  impl_->height = height;
  impl_->width = width;
  return true;
}

void YoloTRTInfer::Postprocess() {
    impl_->detected_objects.clear();
    
    auto num_detections  = output_bindings_[0].dims.d[1];  // 检测框数量，yolo默认300
    auto num_channels = output_bindings_[0].dims.d[2];     // 每个检测框的通道数，相当于矩阵列数

    // === 分割任务：提前设置原型掩码信息 ===
    if (impl_->task_type == YoloTaskType::SEGMENT && output_bindings_.size() >= 2) {
        // 原型掩码 tensor shape: [1, num_proto, proto_h, proto_w] (NCHW)
        g_segment_proto.proto_masks = static_cast<const float*>(host_buffers_[1]);
        g_segment_proto.num_proto   = output_bindings_[1].dims.d[1];
        g_segment_proto.proto_h     = output_bindings_[1].dims.d[2];
        g_segment_proto.proto_w     = output_bindings_[1].dims.d[3];
        g_segment_proto.net_width   = input_binding_.dims.d[3];
        g_segment_proto.net_height  = input_binding_.dims.d[2];
        std::cout << "[Segment] proto_masks loaded: " << g_segment_proto.num_proto
                  << "x" << g_segment_proto.proto_h << "x" << g_segment_proto.proto_w
                  << "  net_size: " << g_segment_proto.net_width << "x" << g_segment_proto.net_height << std::endl;
    }

    // 调用后处理函数
    if (impl_->postprocess_func) {
        impl_->postprocess_func(
            static_cast<float*>(host_buffers_[0]),
            num_channels,
            num_detections,
            impl_->detected_objects);
    }
    
    // 将关键点还原到原图坐标
    for (auto& obj : impl_->detected_objects) {
      for (auto& point : obj.points) {
        point = ScaleCoords(point);
      }
    }
    
    // 将边界框还原到原图坐标
    for (auto& obj : impl_->detected_objects) {
        auto point = ScaleCoords({obj.bbox.x, obj.bbox.y});
        obj.bbox.width /= impl_->resize_ratio;
        obj.bbox.height /= impl_->resize_ratio;
        obj.bbox.x = point.x;
        obj.bbox.y = point.y;
    }
}

cv::Point2f YoloTRTInfer::ScaleCoords(const cv::Point2f& point) {
  float x = (point.x - impl_->dw) / impl_->resize_ratio;
  float y = (point.y - impl_->dh) / impl_->resize_ratio;
  x = std::min(std::max(x, 0.0f), static_cast<float>(impl_->width - 1));
  y = std::min(std::max(y, 0.0f), static_cast<float>(impl_->height - 1));
  return cv::Point2f(x, y);
}

void YoloTRTInfer::ScalePoints(std::vector<cv::Point2f>& points) {
  for (auto& point : points) {
    point = ScaleCoords(point);
  }
}

void YoloTRTInfer::CreateStrategy(YoloTaskType task_type) {
  switch (task_type) {
      case YoloTaskType::DETECT:
          impl_->postprocess_func = DetectPostprocess;
          break;
      case YoloTaskType::OBB:
          impl_->postprocess_func = OBBPostprocess;
          break;
      case YoloTaskType::POSE:
          impl_->postprocess_func = PosePostprocess;
          break;
      case YoloTaskType::SEGMENT:
          impl_->postprocess_func = SegmentPostprocess;
          break;
      default:
          impl_->postprocess_func = DetectPostprocess;
  }
  const char* task_type_str[] = {"detect", "obb", "pose", "segment"};
  std::cout << CV_ALGORITHM_LOG_PREFIX << "[YoloTRTInfer] Create strategy for task type: " << task_type_str[static_cast<int>(task_type)] << std::endl;
}