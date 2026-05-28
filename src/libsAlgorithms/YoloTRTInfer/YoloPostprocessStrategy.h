// /data/Code/tianma_project_ws/ebots_ros2_perception/workspace/ebots_perception_core/vendor/CVAlgorithmRepo/src/libsAlgorithms/YoloTRTInfer/YoloPostprocessStrategy.h
#pragma once

#include <vector>
#include <memory>
#include "YoloTRTInfer.h"

// 目标检测后处理函数
void DetectPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_anchors,
    std::vector<YoloObject>& detected_objects);

// 旋转目标检测后处理函数
void OBBPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_anchors,
    std::vector<YoloObject>& detected_objects);

// 关键点检测后处理函数
void PosePostprocess(
    const float* output_buffer,
    int num_channels,
    int num_anchors,
    std::vector<YoloObject>& detected_objects);

// 分割后处理函数
void SegmentPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_anchors,
    std::vector<YoloObject>& detected_objects);