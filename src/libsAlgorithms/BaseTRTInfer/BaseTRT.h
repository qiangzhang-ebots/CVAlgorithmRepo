#ifndef BASETRT_H
#define BASETRT_H

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
	size_t size = 0;
};

class BASETRTINFER_EXPORT BaseTRT {
 public:
	BaseTRT();
	virtual ~BaseTRT();

	bool LoadModel(const std::string& modelPath);

	/*
		设置运行的gpu_id，需要类初始化时设置。默认为0
	*/
	bool SetGPUId(int gpu_id); 
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
	std::vector<Binding> output_bindings_;
	int gpu_id_ = 0;

	std::vector<void*> device_buffers_;
	void* host_input_buffer_ = nullptr;
	std::vector<void*> host_buffers_;
};

#endif 