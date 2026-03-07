#include "BaseYoloTRTInfer.h"
#include <fstream>
#include <cassert>
#include <stdexcept>

size_t getSizeByDim(const nvinfer1::Dims& dims) {
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];
  }
  return size;
}

size_t getElementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;  // 32-bit float
    case nvinfer1::DataType::kHALF:
      return 2;  // 16-bit float
    case nvinfer1::DataType::kINT8:
      return 1;  // 8-bit int
    case nvinfer1::DataType::kINT32:
      return 4;  // 32-bit int
    case nvinfer1::DataType::kBOOL:
      return 1;  // Boolean
    default:
      throw std::runtime_error("Unknown DataType");
  }
}

BaseYoloTRTInfer::BaseYoloTRTInfer() {}

BaseYoloTRTInfer::~BaseYoloTRTInfer() {}

bool BaseYoloTRTInfer::LoadModel(const std::string& modelPath) {
  std::ifstream file(modelPath, std::ios::binary);
  assert(file.good());
  file.seekg(0, std::ios::end);
  size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  char* trtModelStream = new char[fileSize];
  assert(trtModelStream);
  file.read(trtModelStream, fileSize);
  file.close();

  runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger_));
  assert(runtime_ != nullptr);
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime_->deserializeCudaEngine(trtModelStream, fileSize));
  assert(engine_ != nullptr);
  delete[] trtModelStream;
  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  assert(context_ != nullptr);

  cudaStreamCreate(&stream_);

#if NV_TENSORRT_MAJOR >= 10
  int num_bindings = engine_->getNbIOTensors();
#else
  int num_bindings = engine_->getNbBindings();
#endif

  if (num_bindings != 2) {
    std::cerr << "Expected exactly 2 bindings (input and output), but got "
              << num_bindings << std::endl;
    return false;
  }
  for (int i = 0; i < 2; i++) {
    Binding binding;

#if NV_TENSORRT_MAJOR >= 10
    const char* tensorName = engine_->getIOTensorName(i);
    nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
    nvinfer1::DataType dtype = engine_->getTensorDataType(tensorName);

    binding.name = tensorName;
    binding.dims = dims;
    binding.dtype = dtype;
    binding.size = getSizeByDim(dims) * getElementSize(dtype);

    nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensorName);

    bool IsInput = (mode == nvinfer1::TensorIOMode::kINPUT);
#else
    const char* tensorName = engine_->getBindingName(i);
    nvinfer1::Dims dims = engine_->getBindingDimensions(i);
    nvinfer1::DataType dtype = engine_->getBindingDataType(i);

    binding.name = tensorName;
    binding.dims = dims;
    binding.dtype = dtype;
    binding.size = getSizeByDim(dims) * getElementSize(dtype);

    bool IsInput = engine_->bindingIsInput(i);
#endif

    if (IsInput) {
      input_binding_ = binding;
#if NV_TENSORRT_MAJOR >= 10
      context_->setInputShape(tensorName, dims);  // Set input dimensions
#else
      context_->setBindingDimensions(i, dims);
#endif
    } else {
      output_binding_ = binding;
    }
  }

  if (input_binding_.size == 0 || output_binding_.size == 0) {
    std::cerr << "Failed to determine input/output bindings." << std::endl;
    return false;
  }

  std::cout << "load model success!" << std::endl;
  MakePipe(true);  // Warmup

  return true;
}

void BaseYoloTRTInfer::MakePipe(bool is_warmup) {
  cudaError_t err;

  err = cudaMallocAsync(&device_buffers_[0], input_binding_.size, stream_);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for input tensor "
              << input_binding_.name << std::endl;
    return;
  }

  err = cudaMallocAsync(&device_buffers_[1], output_binding_.size, stream_);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for output tensor "
              << output_binding_.name << std::endl;
    return;
  }
  err = cudaHostAlloc(&host_buffer_, output_binding_.size, 0);
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate host memory for output tensor "
              << output_binding_.name << std::endl;
    return;
  }

  if (is_warmup) {
    for (int i = 0; i < 10; i++) {
      void* h_ptr = malloc(input_binding_.size);
      memset(h_ptr, 0, input_binding_.size);  // Fill with zeros
      err = cudaMemcpyAsync(device_buffers_[0], h_ptr, input_binding_.size,
                            cudaMemcpyHostToDevice, stream_);
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy data to device for input tensor "
                  << input_binding_.name << std::endl;
        return;
      }
      free(h_ptr);
      Infer();
      std::cout << "Warmup iteration " << i + 1 << " completed." << std::endl;
    }
  }
}

bool BaseYoloTRTInfer::Infer() {
  // This is where you would enqueue the inference work using
  // context_->enqueueV3() or similar. You would need to set up the input and
  // output buffers correctly before calling this.
  cudaError_t err;

  bool ret = false;

#if NV_TENSORRT_MAJOR >= 10
  ret = context_->setTensorAddress(input_binding_.name.c_str(),
                                   device_buffers_[0]);
  if (!ret) {
    std::cerr << "Failed to set tensor address for input tensor "
              << input_binding_.name << std::endl;
    return false;
  }
  ret = context_->setTensorAddress(output_binding_.name.c_str(),
                                   device_buffers_[1]);
  if (!ret) {
    std::cerr << "Failed to set tensor address for output tensor "
              << output_binding_.name << std::endl;
    return false;
  }

  ret = context_->enqueueV3(stream_);
#else
  // void* bindings[] = {device_buffers_[0], device_buffers_[1]};
  ret = context_->enqueueV2(device_buffers_, stream_, nullptr);
#endif

  if (!ret) {
    std::cerr << "Failed to enqueue inference for context." << std::endl;
    return false;
  }

  auto osize = output_binding_.size;
  err = cudaMemcpyAsync(host_buffer_, device_buffers_[1], osize,
                        cudaMemcpyDeviceToHost, stream_);

  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data from device for output tensor "
              << output_binding_.name << std::endl;
    return false;
  }

  cudaStreamSynchronize(stream_);
  return true;
}

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
  cv::Mat nchw;

  auto width = input_binding_.dims.d[3];
  auto height = input_binding_.dims.d[2];
  cv::Size size(width, height);

  Letterbox(input_image, nchw,
            size);  // Implement letterbox resizing to fit model input size
  // context_->setInputShape(tensorName, dims);
  cudaError_t err;
  err = cudaMemcpyAsync(device_buffers_[0], nchw.ptr<float>(),
                        nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice,
                        stream_);
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

  float dw = (inp_w - padw) / 2.f;
  float dh = (inp_h - padh) / 2.f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));

  cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  // Convert to RGB and normalize to NCHW manually since dnn::blobFromImage is unavailable
  cv::Mat rgb;
  cv::cvtColor(tmp, rgb, cv::COLOR_BGR2RGB);
  
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // HWC to NCHW
  std::vector<cv::Mat> channels(3);
  cv::split(rgb, channels);

  int sizes[] = {1, 3, size.height, size.width};
  output.create(4, sizes, CV_32F);
  int channel_size = size.width * size.height;
  for (int i = 0; i < 3; ++i) {
    memcpy(output.ptr<float>(0, i), channels[i].ptr<float>(), channel_size * sizeof(float));
  }

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