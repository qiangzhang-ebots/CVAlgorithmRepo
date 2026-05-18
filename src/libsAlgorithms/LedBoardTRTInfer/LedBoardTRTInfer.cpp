#include "LedBoardTRTInfer.h"
#include <cassert>
#include <algorithm>

namespace {
// 数据布局偏移量枚举 - 清晰定义模型输出的数据结构
// 模型输出格式: [x1, y1, x2, y2, score, label, kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...]
enum class OutputOffset {
    X1 = 0,
    Y1 = 1,
    X2 = 2,
    Y2 = 3,
    SCORE = 4,
    LABEL = 5,
    KEYPOINTS_START = 6  // 关键点数据起始位置
};

// 单个关键点的数据大小（x, y, confidence）
constexpr int KEYPOINT_SIZE = 3;
}

LedBoardTRTInfer::LedBoardTRTInfer() {}
LedBoardTRTInfer::~LedBoardTRTInfer() {}

void LedBoardTRTInfer::Postprocess() {
    detected_objects_.clear();
    BaseYoloTRTInfer::Postprocess();
}

void LedBoardTRTInfer::PostprocessOneObject(const float* row_ptr) {
    // 安全检查：确保输入指针有效
    if (!row_ptr) {
        return;
    }

    LedBoardObject obj;

    // 解析置信度并进行阈值过滤
    float score = row_ptr[static_cast<int>(OutputOffset::SCORE)];
    if (score <= 0.5f) {
        return;
    }

    // 解析边界框坐标
    float x1 = row_ptr[static_cast<int>(OutputOffset::X1)];
    float y1 = row_ptr[static_cast<int>(OutputOffset::Y1)];
    float x2 = row_ptr[static_cast<int>(OutputOffset::X2)];
    float y2 = row_ptr[static_cast<int>(OutputOffset::Y2)];

    // 将坐标从模型输入尺寸转换回原图尺寸
    cv::Point2f p1 = ScaleCoords(cv::Point2f(x1, y1));
    cv::Point2f p2 = ScaleCoords(cv::Point2f(x2, y2));

    // 构建边界框（最小外接矩）
    obj.bounding_box.x = p1.x;
    obj.bounding_box.y = p1.y;
    obj.bounding_box.width = p2.x - p1.x;
    obj.bounding_box.height = p2.y - p1.y;

    // 解析类别标签
    int label = static_cast<int>(row_ptr[static_cast<int>(OutputOffset::LABEL)]);
    obj.label = label;
    obj.box_confidence = score;

    // 根据标签获取类别名称
    auto name_it = LED_BOARD_CLASS_NAMES.find(label);
    if (name_it != LED_BOARD_CLASS_NAMES.end()) {
        obj.class_name = name_it->second;
    } else {
        obj.class_name = "Unknown";
    }

    // 根据类别获取关键点数量
    int num_keypoints = 0;
    auto kp_num_it = LED_BOARD_CLASS_KEYPOINT_NUM.find(label);
    if (kp_num_it != LED_BOARD_CLASS_KEYPOINT_NUM.end()) {
        num_keypoints = kp_num_it->second;
    } else {
        return;  // 未知类别，跳过
    }

    // 解析关键点数据
    int kp_start_idx = static_cast<int>(OutputOffset::KEYPOINTS_START);
    obj.keypoints.reserve(num_keypoints);
    obj.keypoint_confidences.reserve(num_keypoints);

    for (int k = 0; k < num_keypoints; ++k) {
        int base_idx = kp_start_idx + k * KEYPOINT_SIZE;
        float kpx = row_ptr[base_idx];
        float kpy = row_ptr[base_idx + 1];
        float kp_confidence = row_ptr[base_idx + 2];

        // 将关键点坐标转换回原图尺寸
        cv::Point2f kp = ScaleCoords(cv::Point2f(kpx, kpy));

        // 边界裁剪
        kpx = std::max(0.0f, std::min(kp.x, static_cast<float>(params_.width)));
        kpy = std::max(0.0f, std::min(kp.y, static_cast<float>(params_.height)));

        obj.keypoints.emplace_back(kpx, kpy);
        obj.keypoint_confidences.push_back(kp_confidence);
    }

    detected_objects_.push_back(obj);
}

std::vector<LedBoardObject> LedBoardTRTInfer::GetObjects() {
    return detected_objects_;
}