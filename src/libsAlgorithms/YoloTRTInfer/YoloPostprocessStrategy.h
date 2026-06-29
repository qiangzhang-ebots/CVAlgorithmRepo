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

// 分割后处理需要的原型掩码信息
// 在调用 SegmentPostprocess 前由调用方设置
struct SegmentProtoInfo {
    const float* proto_masks = nullptr;  // [num_proto, proto_h, proto_w] NCHW
    int num_proto = 0;
    int proto_h = 0;
    int proto_w = 0;
    // 模型输入尺寸（letterbox后的尺寸）
    int net_width = 0;
    int net_height = 0;
};
extern SegmentProtoInfo g_segment_proto;