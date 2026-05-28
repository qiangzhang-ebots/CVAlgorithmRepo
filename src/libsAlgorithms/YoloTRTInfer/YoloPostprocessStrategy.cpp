// /data/Code/tianma_project_ws/ebots_ros2_perception/workspace/ebots_perception_core/vendor/CVAlgorithmRepo/src/libsAlgorithms/YoloTRTInfer/YoloPostprocessStrategy.cpp
#include "YoloPostprocessStrategy.h"


// 目标检测后处理 - 返回 x1,y1,x2,y2
void DetectPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects) {
    detected_objects.clear();
    
    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;// 每个detection的输出数据
        
        // 假设输出格式: x1,y1,x2,y2,conf,cls
        float conf = data[4];
        if (conf < 0.5f) continue;  // 置信度过滤

        float x1 = data[0];
        float y1 = data[1];
        float x2 = data[2];
        float y2 = data[3];
        int label = static_cast<int>(round(data[5]));
        
        YoloObject obj;
        obj.label = label;
        obj.confidence = conf;
        obj.points.emplace_back(x1, y1);  // x1, y1
        obj.points.emplace_back(x2, y2);  // x2, y2
        obj.bbox = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        
        detected_objects.push_back(obj);
    }
}

auto get_bounding_box = [](const std::vector<cv::Point2f> &points) -> cv::Rect2f {
    if (points.empty()) {
        return cv::Rect2f();
    }
    float min_x = points[0].x, max_x = points[0].x;
    float min_y = points[0].y, max_y = points[0].y;
    for (const auto& p : points) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
};

// 旋转目标检测后处理 - 返回4个点
void OBBPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects) {

    detected_objects.clear();
    if (num_channels < 10) {    // 所需最小通道数 = 10
        printf("[OBBPostprocess] Invalid num_channels=%d, expected at least 10\n", num_channels);
        return;
    }

    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;

        // 格式：[cx, cy, w, h, r, conf, cls] + x1 y1 x2 y2 x3 y3 x4 y4
        float cx = data[0];
        float cy = data[1];
        float w = data[2];
        float h = data[3];
        float r = data[4];
        float obj_conf = data[5];
        int label = static_cast<int>(round(data[6]));

        // 置信度过滤
        if (obj_conf < 0.5f) continue;

        float x1 = data[7];
        float y1 = data[8];
        float x2 = data[9];
        float y2 = data[10];
        float x3 = data[11];
        float y3 = data[12];
        float x4 = data[13];
        float y4 = data[14];
        
        YoloObject obj;
        obj.label = label;
        obj.confidence = obj_conf;
        obj.points.emplace_back(x1, y1);  // 点1
        obj.points.emplace_back(x2, y2);  // 点2
        obj.points.emplace_back(x3, y3);  // 点3
        obj.points.emplace_back(x4, y4);  // 点4

        obj.bbox = get_bounding_box(obj.points);

        detected_objects.push_back(obj);
    }
}

// 关键点检测后处理 - 返回N个关键点
void PosePostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects){
    detected_objects.clear();
    // 计算关键点数量 K, 模型输出的原始个数，这个似乎不受训练类别数影响
    const int K = (num_channels - 6) / 3;
    
    // 验证通道数是否合法
    if (num_channels < 6 || (num_channels - 6) % 3 != 0 || K <= 0) {
        printf("[PosePostprocess] Invalid num_channels=%d, expected format: cls x y w h kx1 ky1 c1 ...\n", num_channels);
        return;
    }

    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;

        // 格式：x1 y1 x2 y2 score class k1x ky1 v1 k2x ky2 v2 ...
        // 1. 读 xyxy + 置信度 + 类别
        float x1 = data[0];
        float y1 = data[1];
        float x2 = data[2];
        float y2 = data[3];
        float obj_conf = data[4];
        int label = static_cast<int>(round(data[5]));

        // 置信度过滤
        if (obj_conf < 0.5f) continue;// 置信度过滤，置信度一定在[0,1]，超过关键点数量后，后面的占位字符都是乱的，不能被纳入

        YoloObject obj;
        obj.label = label;
        obj.confidence = obj_conf;

        // 关键点：kx ky v。因为不知道实际关键点的数量，所以这里直接按 K 个关键点处理，在调用层因为知道关键点格式，再剔除后面的占位字符
        for (int k = 0; k < K; ++k) {
            int idx = 6 + k*3;
            float kx = data[idx];
            float ky = data[idx+1];
            float kv = data[idx+2];
            obj.points.emplace_back(kx, ky);
            obj.point_confidences.push_back(kv);
        }

        // 计算边界框
        obj.bbox = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2)); // 模型给出的 x1, y1 是左上角，x2, y2 是右下角

        detected_objects.push_back(obj);
    }
}

// 分割后处理 - 返回轮廓点集，待验证
void SegmentPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects) {
    detected_objects.clear();
    const int MASK_POINT_COUNT = 32; // 32个值 = 16个(x,y)点

    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;

        float x1 = data[0];
        float y1 = data[1];
        float x2 = data[2];
        float y2 = data[3];
        float conf = data[4];
        int label = static_cast<int>(round(data[5]));

        if (conf < 0.5f) continue;

        YoloObject obj;
        obj.label = label;
        obj.confidence = conf;
        obj.bbox = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));

        // 掩码转换
        const float* mask_data = data + 6; // 相对 bbox的轮廓点，每个点占2个值（x,y），相对bbox 0-1归一化
        float box_w = obj.bbox.width;
        float box_h = obj.bbox.height;
        float box_x = obj.bbox.x;
        float box_y = obj.bbox.y;

        // 32个数值 → 16个 (x,y) 轮廓点
        for (int j = 0; j < MASK_POINT_COUNT; j += 2) {
            float nx = mask_data[j];
            float ny = mask_data[j+1];

            // 直接转成 推理图上的绝对像素坐标
            float x = box_x + nx * box_w;
            float y = box_y + ny * box_h;

            obj.points.emplace_back(x, y);
        }

        detected_objects.push_back(obj);
    }
}