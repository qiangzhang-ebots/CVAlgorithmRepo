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

bool SetCudaDevice(int gpu_id, const char* stage) {
	cudaError_t err = cudaSetDevice(gpu_id);
	if (err != cudaSuccess) {
		std::cerr << "Failed to set CUDA device to gpu_id=" << gpu_id << " at "
						<< stage << ": " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}

}  // namespace

BaseTRT::BaseTRT() {}

BaseTRT::~BaseTRT() {
	if (!SetCudaDevice(gpu_id_, "BaseTRT::~BaseTRT")) {
		return;
	}

	if (stream_ != nullptr) {
		cudaStreamSynchronize(stream_);
	}
	if (device_buffers_[0] != nullptr) {
		cudaFree(device_buffers_[0]);
	}
	if (device_buffers_[1] != nullptr) {
		cudaFree(device_buffers_[1]);
	}
	if (host_input_buffer_ != nullptr) {
		cudaFreeHost(host_input_buffer_);
	}
	if (host_buffer_ != nullptr) {
		cudaFreeHost(host_buffer_);
	}
	if (stream_ != nullptr) {
		cudaStreamDestroy(stream_);
	}
}

bool BaseTRT::LoadModel(const std::string& modelPath) 
{
	if (!SetCudaDevice(gpu_id_, "BaseTRT::LoadModel")) {
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

	cudaStreamCreate(&stream_);

	int num_bindings = engine_->getNbIOTensors();
	if (num_bindings != 2) {
		std::cerr << "Expected exactly 2 bindings (input and output), but got "
							<< num_bindings << std::endl;
		return false;
	}
	for (int i = 0; i < 2; i++) {
		Binding binding;

		const char* tensorName = engine_->getIOTensorName(i);
		nvinfer1::Dims dims = engine_->getTensorShape(tensorName);
		nvinfer1::DataType dtype = engine_->getTensorDataType(tensorName);

		binding.name = tensorName;
		binding.dims = dims;
		binding.dtype = dtype;
		binding.size = GetSizeByDim(dims) * GetElementSize(dtype);

		nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(tensorName);

		bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);
		if (is_input) {
			input_binding_ = binding;
			context_->setInputShape(tensorName, dims);
		} else {
			output_binding_ = binding;
		}
	}

	if (input_binding_.size == 0 || output_binding_.size == 0) {
		std::cerr << "Failed to determine input/output bindings." << std::endl;
		return false;
	}

	std::cout << "load model success!" << std::endl;
	MakePipe(true);

	return true;
}

bool BaseTRT::SetGPUId(int gpu_id)
{
	gpu_id_ = gpu_id;
  return SetCudaDevice(gpu_id_, "BaseTRT::~BaseTRT");
}

void BaseTRT::MakePipe(bool is_warmup) {
	if (!SetCudaDevice(gpu_id_, "BaseTRT::MakePipe")) {
		return;
	}

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
	err = cudaHostAlloc(&host_input_buffer_, input_binding_.size, 0);
	if (err != cudaSuccess) {
		std::cerr << "Failed to allocate host memory for input tensor "
							<< input_binding_.name << std::endl;
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
			void* host_ptr = malloc(input_binding_.size);
			memset(host_ptr, 0, input_binding_.size);
			err = cudaMemcpyAsync(device_buffers_[0], host_ptr, input_binding_.size,
														cudaMemcpyHostToDevice, stream_);
			if (err != cudaSuccess) {
				std::cerr << "Failed to copy data to device for input tensor "
									<< input_binding_.name << std::endl;
				free(host_ptr);
				return;
			}
			free(host_ptr);
			Infer();
			std::cout << "Warmup iteration " << i + 1 << " completed." << std::endl;
		}
	}
}

bool BaseTRT::Infer() {
	if (!SetCudaDevice(gpu_id_, "BaseTRT::Infer")) {
		return false;
	}

	cudaError_t err;

	bool ret = context_->setTensorAddress(input_binding_.name.c_str(),
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
	if (!ret) {
		std::cerr << "Failed to enqueue inference for context." << std::endl;
		return false;
	}

	err = cudaMemcpyAsync(host_buffer_, device_buffers_[1], output_binding_.size,
												cudaMemcpyDeviceToHost, stream_);

	if (err != cudaSuccess) {
		std::cerr << "Failed to copy data from device for output tensor "
							<< output_binding_.name << std::endl;
		return false;
	}

	cudaStreamSynchronize(stream_);
	return true;
}
