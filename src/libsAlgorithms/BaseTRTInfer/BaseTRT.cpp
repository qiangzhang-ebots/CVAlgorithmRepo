#include "BaseTRT.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

size_t GetSizeByDim(const nvinfer1::Dims& dims) {
	size_t size = 1;
	for (int i = 0; i < dims.nbDims; ++i) {
		size *= dims.d[i];
	}
	return size;
}

size_t GetElementSize(nvinfer1::DataType dtype) {
	switch (dtype) {
		case nvinfer1::DataType::kFLOAT:
			return 4;
		case nvinfer1::DataType::kHALF:
			return 2;
		case nvinfer1::DataType::kINT8:
			return 1;
		case nvinfer1::DataType::kINT32:
			return 4;
		case nvinfer1::DataType::kBOOL:
			return 1;
		default:
			throw std::runtime_error("Unknown DataType");
	}
}

}  // namespace

BaseTRT::BaseTRT() {}

bool BaseTRT::SetCudaDevice(const char* stage) {
	cudaError_t err = cudaSetDevice(gpu_id_);
	if (err != cudaSuccess) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to set CUDA device to gpu_id=" << gpu_id_ << " at "
							<< stage << ": " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}

BaseTRT::~BaseTRT() {
	if (!SetCudaDevice("BaseTRT::~BaseTRT")) {
		return;
	}

	if (stream_ != nullptr) {
		cudaStreamSynchronize(stream_);
	}
	for (auto* buf : device_buffers_) {
		if (buf != nullptr) {
			cudaFree(buf);
		}
	}
	if (host_input_buffer_ != nullptr) {
		cudaFreeHost(host_input_buffer_);
	}
	for (auto* buf : host_buffers_) {
		if (buf != nullptr) {
			cudaFreeHost(buf);
		}
	}
	if (stream_ != nullptr) {
		cudaStreamDestroy(stream_);
	}
}

bool BaseTRT::LoadModel(const std::string& modelPath) 
{
	if (!SetCudaDevice("BaseTRT::LoadModel")) {
		return false;
	}

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

	cudaError_t stream_err = cudaStreamCreate(&stream_);
	if (stream_err != cudaSuccess) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to create CUDA stream: "
							<< cudaGetErrorString(stream_err) << std::endl;
		return false;
	}

	int num_bindings = engine_->getNbIOTensors();
	if (num_bindings < 2) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Expected at least 2 bindings (input and output), but got "
							<< num_bindings << std::endl;
		return false;
	}
	input_binding_ = Binding();
	output_bindings_.clear();

	for (int i = 0; i < num_bindings; i++) {
		Binding binding;

		const char* tensorName = engine_->getIOTensorName(i);
		nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
		nvinfer1::DataType dtype = engine_->getTensorDataType(tensorName);

		std::cout << CV_ALGORITHM_LOG_PREFIX << "tensorName " << tensorName << " dims: [ ";
		for (int j = 0; j < dims.nbDims; ++j) { std::cout << dims.d[j] << " "; }
		std::cout << "]" << std::endl;

		binding.name = tensorName;
		binding.dims = dims;
		binding.dtype = dtype;
		binding.size = GetSizeByDim(dims) * GetElementSize(dtype);

		nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensorName);

		if (mode == nvinfer1::TensorIOMode::kINPUT) {
			input_binding_ = binding;
			context_->setInputShape(tensorName, dims);
		} else {
			output_bindings_.push_back(binding);
		}
	}

	if (input_binding_.size == 0 || output_bindings_.empty()) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to determine input/output bindings." << std::endl;
		return false;
	}

	std::cout << CV_ALGORITHM_LOG_PREFIX << "load model success! [" << output_bindings_.size() << " output(s)]" << std::endl;
	MakePipe(true);

	return true;
}

bool BaseTRT::SetGPUId(int gpu_id)
{
	gpu_id_ = gpu_id;
  	return SetCudaDevice("BaseTRT::SetGPUId");
}

void BaseTRT::MakePipe(bool is_warmup) {
	if (!SetCudaDevice("BaseTRT::MakePipe")) {
		return;
	}

	cudaError_t err;

	// 释放旧 buffer（如果有）
	for (auto* buf : device_buffers_) { if (buf) cudaFree(buf); }
	for (auto* buf : host_buffers_) { if (buf) cudaFreeHost(buf); }
	device_buffers_.clear();
	host_buffers_.clear();

	// input dev & host
	void* dev_buf = nullptr;
	err = cudaMallocAsync(&dev_buf, input_binding_.size, stream_);
	if (err != cudaSuccess) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to allocate device memory for input tensor "
							<< input_binding_.name << ": " << cudaGetErrorString(err)
							<< std::endl;
		return;
	}
	device_buffers_.push_back(dev_buf);

	err = cudaHostAlloc(&host_input_buffer_, input_binding_.size, 0);
	if (err != cudaSuccess) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to allocate host memory for input tensor "
							<< input_binding_.name << ": " << cudaGetErrorString(err)
							<< std::endl;
		return;
	}

	// output dev & host (for each output)
	for (size_t i = 0; i < output_bindings_.size(); i++) {
		void* dev_out = nullptr;
		err = cudaMallocAsync(&dev_out, output_bindings_[i].size, stream_);
		if (err != cudaSuccess) {
			std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to allocate device memory for output tensor "
								<< output_bindings_[i].name << ": " << cudaGetErrorString(err)
								<< std::endl;
			return;
		}
		device_buffers_.push_back(dev_out);

		void* host_out = nullptr;
		err = cudaHostAlloc(&host_out, output_bindings_[i].size, 0);
		if (err != cudaSuccess) {
			std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to allocate host memory for output tensor "
								<< output_bindings_[i].name << ": " << cudaGetErrorString(err)
								<< std::endl;
			return;
		}
		host_buffers_.push_back(host_out);
	}

	if (is_warmup) {
		for (int i = 0; i < 10; i++) {
			void* host_ptr = malloc(input_binding_.size);
			memset(host_ptr, 0, input_binding_.size);
			err = cudaMemcpyAsync(device_buffers_[0], host_ptr, input_binding_.size,
														cudaMemcpyHostToDevice, stream_);
			if (err != cudaSuccess) {
				std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to copy data to device for input tensor "
									<< input_binding_.name << ": " << cudaGetErrorString(err)
									<< std::endl;
				free(host_ptr);
				return;
			}
			free(host_ptr);
			Infer();
			// std::cout << CV_ALGORITHM_LOG_PREFIX << "Warmup iteration " << i + 1 << " completed." << std::endl;
		}
	}
}

bool BaseTRT::Infer() {
	if (!SetCudaDevice("BaseTRT::Infer")) {
		return false;
	}

	cudaError_t err;

	// input tensor address
	bool ret = context_->setTensorAddress(input_binding_.name.c_str(),
																				device_buffers_[0]);
	if (!ret) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to set tensor address for input tensor "
							<< input_binding_.name << std::endl;
		return false;
	}
	// output tensor addresses (device_buffers_[0] = input, device_buffers_[1..N] = outputs)
	for (size_t i = 0; i < output_bindings_.size(); i++) {
		ret = context_->setTensorAddress(output_bindings_[i].name.c_str(),
																		 device_buffers_[1 + i]);
		if (!ret) {
			std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to set tensor address for output tensor "
								<< output_bindings_[i].name << std::endl;
			return false;
		}
	}

	ret = context_->enqueueV3(stream_);
	if (!ret) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to enqueue inference for context." << std::endl;
		return false;
	}

	// D2H copy for all outputs
	for (size_t i = 0; i < output_bindings_.size(); i++) {
		err = cudaMemcpyAsync(host_buffers_[i], device_buffers_[1 + i],
													output_bindings_[i].size,
													cudaMemcpyDeviceToHost, stream_);
		if (err != cudaSuccess) {
			std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to copy data from device for output tensor "
								<< output_bindings_[i].name << ": " << cudaGetErrorString(err)
								<< std::endl;
			return false;
		}
	}

	err = cudaStreamSynchronize(stream_);
	if (err != cudaSuccess) {
		std::cerr << CV_ALGORITHM_LOG_PREFIX << "Failed to synchronize CUDA stream: "
							<< cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}
