#include "YoloPostprocessStrategy.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

// 原型掩码信息（在调用前由 YoloTRTInfer::Postprocess 设置）
SegmentProtoInfo g_segment_proto;

// 通用 NMS 函数。暂时没有调用
static void nms(std::vector<YoloObject>& objects, float iou_threshold = 0.45f) {
    if (objects.empty()) return;
    
    // Lambda: 计算 IoU
    auto iou = [](const cv::Rect2f& a, const cv::Rect2f& b) {
        cv::Rect2f inter = a & b;
        if (inter.width <= 0 || inter.height <= 0) return 0.0f;
        float inter_area = inter.width * inter.height;
        float area_a = a.width * a.height;
        float area_b = b.width * b.height;
        return inter_area / (area_a + area_b - inter_area);
    };
    
    // 按置信度降序排序
    std::sort(objects.begin(), objects.end(), [](const YoloObject& a, const YoloObject& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<bool> keep(objects.size(), true);
    
    for (size_t i = 0; i < objects.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (!keep[j]) continue;
            // 使用 Lambda 计算 IoU
            float iou_val = iou(objects[i].bbox, objects[j].bbox);
            if (iou_val > iou_threshold) {
                keep[j] = false;
            }
        }
    }
    std::vector<YoloObject> result;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (keep[i]) {
            result.push_back(objects[i]);
        }
    }
    objects = std::move(result);
}

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

        std::cout << CV_ALGORITHM_LOG_PREFIX << "label " << label << " conf " << conf <<  "points [" << x1 << "," << y1 << "] [" << x2 << "," << y2 << "]\n";
        
        YoloObject obj;
        obj.label = label;
        obj.confidence = conf;
        obj.points.emplace_back(x1, y1);  // x1, y1
        obj.points.emplace_back(x2, y2);  // x2, y2
        obj.bbox = cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        
        detected_objects.push_back(obj);
    }
    // NMS
    //nms(detected_objects);
}

// 旋转目标检测后处理 - 返回4个点
void OBBPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects) {

    detected_objects.clear();
    if (num_channels < 7) {    // 所需最小通道数 = 7
        printf("[OBBPostprocess] Invalid num_channels=%d, expected at least 7\n", num_channels);
        return;
    }

    // 计算旋转矩形四个角点
       auto obb_points = [](float cx, float cy, float w, float h, float r) {
        // 半宽、半高
        float half_w = w * 0.5f;
        float half_h = h * 0.5f;
        // 三角函数
        float cos_r = std::cos(r);
        float sin_r = std::sin(r);
        // 旋转矩形四个角点相对中心的偏移量
        float dx1 = -half_w * cos_r - (-half_h) * sin_r;
        float dy1 = -half_w * sin_r + (-half_h) * cos_r;
        float dx2 = half_w * cos_r - (-half_h) * sin_r;
        float dy2 = half_w * sin_r + (-half_h) * cos_r;
        float dx3 = half_w * cos_r - half_h * sin_r;
        float dy3 = half_w * sin_r + half_h * cos_r;
        float dx4 = -half_w * cos_r - half_h * sin_r;
        float dy4 = -half_w * sin_r + half_h * cos_r;
        return std::make_tuple(
            cx + dx1, cy + dy1, cx + dx2, cy + dy2, cx + dx3, cy + dy3, cx + dx4, cy + dy4
        );
    };

    // 计算最小外接矩形
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

    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;

        // 格式：[cx, cy, w, h, r, conf, cls] + x1 y1 x2 y2 x3 y3 x4 y4
        float cx = data[0];
        float cy = data[1];
        float w = data[2];
        float h = data[3];
        float obj_conf = data[4];
        int label = static_cast<int>(round(data[5])); 
        float r = data[6];

        // 置信度过滤
        if (obj_conf < 0.5f) continue;
        // obb输出只有以上7个通道。所以四个点需要从cx cy w h r计算
        auto [x1, y1, x2, y2, x3, y3, x4, y4] = obb_points(cx, cy, w, h, r);
        
        YoloObject obj;
        obj.label = label;
        obj.confidence = obj_conf;
        obj.points.emplace_back(x1, y1);  // 点1
        obj.points.emplace_back(x2, y2);  // 点2
        obj.points.emplace_back(x3, y3);  // 点3
        obj.points.emplace_back(x4, y4);  // 点4

        obj.bbox = get_bounding_box(obj.points);// 浮点rect

        detected_objects.push_back(obj);
    }

    // NMS
    //nms(detected_objects);
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
    // NMS
    //nms(detected_objects);
}

// 分割后处理 - 使用掩码系数 × 原型掩码 解码分割轮廓
void SegmentPostprocess(
    const float* output_buffer,
    int num_channels,
    int num_detections,
    std::vector<YoloObject>& detected_objects) {
    detected_objects.clear();

    const int NUM_MASK = 32;  // YOLOv8-seg 固定 32 个掩码系数
    int expected_channels = 6 + NUM_MASK;  // xyxy + conf + cls + 32 coeffs

    // 验证通道数
    if (num_channels < expected_channels) {
        printf("[SegmentPostprocess] Invalid num_channels=%d, expected >= %d\n",
               num_channels, expected_channels);
        return;
    }

    // 验证原型掩码信息
    if (!g_segment_proto.proto_masks || g_segment_proto.num_proto != NUM_MASK) {
        printf("[SegmentPostprocess] Missing or invalid prototype masks (num_proto=%d, expected %d)\n",
               g_segment_proto.num_proto, NUM_MASK);
        return;
    }

    int proto_h = g_segment_proto.proto_h;
    int proto_w = g_segment_proto.proto_w;
    int proto_area = proto_h * proto_w;

    // 网络输入尺寸（letterbox后）
    float net_w = static_cast<float>(g_segment_proto.net_width);
    float net_h = static_cast<float>(g_segment_proto.net_height);

    // 原型掩码 → 网络输入图的缩放比例
    float scale_x = net_w / proto_w;
    float scale_y = net_h / proto_h;

    for (int i = 0; i < num_detections; ++i) {
        const float* data = output_buffer + i * num_channels;

        float x1 = data[0];
        float y1 = data[1];
        float x2 = data[2];
        float y2 = data[3];
        float conf = data[4];
        int label = static_cast<int>(round(data[5]));

        if (conf < 0.5f) continue;

        // 读取 32 个掩码系数
        const float* coeffs = data + 6;

        // === 计算原始掩码 ===
        // raw_mask[ph, pw] = Σ coeff[c] * proto[c, ph, pw]
        // 使用 cv::Mat 加速计算
        cv::Mat raw_mask(proto_h, proto_w, CV_32FC1, cv::Scalar(0.0f));
        for (int c = 0; c < NUM_MASK; ++c) {
            if (std::abs(coeffs[c]) < 1e-6f) continue;
            const float* proto_c = g_segment_proto.proto_masks + c * proto_area;
            for (int ph = 0; ph < proto_h; ++ph) {
                float* row = raw_mask.ptr<float>(ph);
                for (int pw = 0; pw < proto_w; ++pw) {
                    row[pw] += coeffs[c] * proto_c[ph * proto_w + pw];
                }
            }
        }

        // sigmoid
        cv::Mat mask;
        cv::exp(-raw_mask, mask);
        mask = 1.0f / (1.0f + mask);

        // === 在原型空间裁剪 bbox 区域 ===
        float px1 = x1 / scale_x;
        float py1 = y1 / scale_y;
        float px2 = x2 / scale_x;
        float py2 = y2 / scale_y;

        // 裁切边界
        int ix1 = std::max(0, static_cast<int>(std::floor(px1)));
        int iy1 = std::max(0, static_cast<int>(std::floor(py1)));
        int ix2 = std::min(proto_w, static_cast<int>(std::ceil(px2)));
        int iy2 = std::min(proto_h, static_cast<int>(std::ceil(py2)));

        // 裁剪 + 缩放到 bbox 大小
        cv::Mat cropped = mask(cv::Range(iy1, iy2), cv::Range(ix1, ix2));
        int box_w = static_cast<int>(x2 - x1 + 0.5f);
        int box_h = static_cast<int>(y2 - y1 + 0.5f);
        if (box_w < 1) box_w = 1;
        if (box_h < 1) box_h = 1;

        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(box_w, box_h));

        // 二值化
        cv::Mat binary;
        cv::threshold(resized, binary, 0.5f, 255, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8UC1);

        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) continue;

        // 取最大轮廓
        auto largest = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });

        // 轮廓点在 resized 空间，resized = (box_w, box_h) 对应网络输入图 bbox
        YoloObject obj;
        obj.label = label;
        obj.confidence = conf;
        obj.bbox = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));

        for (const auto& pt : *largest) {
            obj.points.emplace_back(x1 + pt.x, y1 + pt.y);
        }

        detected_objects.push_back(obj);
    }

    // 应用 NMS
    //nms(detected_objects);
}