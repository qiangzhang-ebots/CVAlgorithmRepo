#ifndef BASETRT_H
#define BASETRT_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>

#include "BaseTRTGlobal.h"

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
	void* buffer;
	int size = 0;
};

class BASETRTINFER_EXPORT BaseTRT {
 public:
	BaseTRT();
	virtual ~BaseTRT();

	bool LoadModel(const std::string& modelPath);

 protected:
	void MakePipe(bool is_warmup = false);
	bool Infer();

 protected:
	std::shared_ptr<nvinfer1::IRuntime> runtime_;
	std::shared_ptr<nvinfer1::ICudaEngine> engine_;
	std::shared_ptr<nvinfer1::IExecutionContext> context_;
	cudaStream_t stream_ = nullptr;
	Logger logger_;

	Binding input_binding_;
	Binding output_binding_;

	void* device_buffers_[2] = {nullptr, nullptr};
	void* host_input_buffer_ = nullptr;
	void* host_buffer_ = nullptr;
};

#endif 