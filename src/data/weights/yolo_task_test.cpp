#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <map>
#include <opencv2/opencv.hpp>
#include "YoloTRTInfer.h"

#include "json.hpp"
using json = nlohmann::json;

std::map<std::string, int> label_map = {
    // detect / obb
    {"coil",        0},
    {"slot_1st",    1},
    {"slot_2nd",    2},
    
    // segment
    {"cable1",      0},
    {"cable2",      1},
    {"cable3",      2},
    {"cable4",      3},
    {"cable5",      4},
    {"cable6",      5},
    
    // pose
    {"1",           0},
    {"2",           1},
    {"3",           2},
    {"4",           3},
    {"5",           4},
    {"6",           5}
};

struct Annotation {
    std::string label;
    std::vector<cv::Point2f> points;
    cv::Rect2f bbox;
};

std::vector<Annotation> ReadAnnotations(const std::string& json_path, int& img_width, int& img_height)
{
    std::ifstream ifs(json_path);
    json js = json::parse(ifs);
    img_width = js["imageWidth"].get<int>();
    img_height = js["imageHeight"].get<int>();

    std::vector<Annotation> res;
    for(auto& shape : js["shapes"])
    {
        Annotation ann;
        ann.label = shape["label"].get<std::string>();
        auto pts_arr = shape["points"];
        for(auto& pt : pts_arr)
        {
            float x = pt[0].get<float>();
            float y = pt[1].get<float>();
            ann.points.emplace_back(x,y);
        }
        // 求外接包围盒
        auto& pts = ann.points;
        float minx=pts[0].x,maxx=pts[0].x;
        float miny=pts[0].y,maxy=pts[0].y;
        for(auto& p:pts){
            minx=std::min(minx,p.x);maxx=std::max(maxx,p.x);
            miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);
        }
        ann.bbox = cv::Rect2f(minx,miny,maxx-minx,maxy-miny);
        res.push_back(ann);
    }
    return res;
}

// 计算关键点平均距离误差
float CalculatePointError(const std::vector<cv::Point2f>& pred, const std::vector<cv::Point2f>& gt) {
    if (pred.empty() || gt.empty()) return 1e9;
    
    float total_dist = 0.0f;
    int count = std::min(pred.size(), gt.size());
    
    for (int i = 0; i < count; i++) {
        float dx = pred[i].x - gt[i].x;
        float dy = pred[i].y - gt[i].y;
        total_dist += std::sqrt(dx * dx + dy * dy);
    }
    
    return total_dist / count;
}

// 将float点转换为int点（用于cv::fillPoly/cv::drawContours）
std::vector<cv::Point> ptsToInt(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point> result;
    result.reserve(pts.size());
    for (const auto& p : pts) {
        result.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    }
    return result;
}

// 计算分割掩码IoU
float CalculateMaskIoU(const std::vector<cv::Point2f>& pred_pts, const std::vector<cv::Point2f>& gt_pts, int img_width, int img_height) {
    if (pred_pts.size() < 3 || gt_pts.size() < 3) return 0.0f;

    cv::Mat pred_mask = cv::Mat::zeros(img_height, img_width, CV_8UC1);
    cv::Mat gt_mask = cv::Mat::zeros(img_height, img_width, CV_8UC1);

    std::vector<std::vector<cv::Point>> pred_contour = {ptsToInt(pred_pts)};
    std::vector<std::vector<cv::Point>> gt_contour = {ptsToInt(gt_pts)};

    cv::fillPoly(pred_mask, pred_contour, cv::Scalar(255));
    cv::fillPoly(gt_mask, gt_contour, cv::Scalar(255));

    cv::Mat intersection, union_mat;
    cv::bitwise_and(pred_mask, gt_mask, intersection);
    cv::bitwise_or(pred_mask, gt_mask, union_mat);

    float inter_area = static_cast<float>(cv::countNonZero(intersection));
    float union_area = static_cast<float>(cv::countNonZero(union_mat));

    if (union_area < 1e-6f) return 0.0f;
    return inter_area / union_area;
}

// 关键点颜色配置 
static const std::vector<cv::Scalar> POINT_COLORS = { 
    cv::Scalar(0, 0, 255),     // 1点：红 
    cv::Scalar(0, 255, 0),     // 2点：绿 
    cv::Scalar(255, 0, 0),     // 3点：蓝 
    cv::Scalar(0, 255, 255),   // 4点：黄 
    cv::Scalar(255, 255, 0),   // 5点：青 
    cv::Scalar(255, 0, 255),   // 6点：紫 
    cv::Scalar(0, 165, 255),   // 7点：橙 
    cv::Scalar(203, 192, 255), // 8点+：粉 
}; 

// 类别颜色配置 
static const std::vector<cv::Scalar> CLASS_COLORS = { 
    cv::Scalar(0, 255, 255),   // 类别0：黄 
    cv::Scalar(255, 0, 255),   // 类别1：紫 
    cv::Scalar(0, 255, 0),     // 类别2：绿 
    cv::Scalar(255, 128, 0),   // 类别3：橙 
    cv::Scalar(128, 0, 255),   // 类别4：深紫 
    cv::Scalar(255, 255, 0),   // 类别5：青 
    cv::Scalar(0, 128, 255),   // 类别6：浅蓝 
    cv::Scalar(128, 255, 0),   // 类别7：浅绿 
}; 


// 绘制检测结果
void DrawResults(cv::Mat& image, const std::vector<YoloObject>& objects, const std::string& task_name) {
    // 绘制预测结果
    for (const auto& obj : objects) {
        cv::Scalar box_color = CLASS_COLORS[obj.label%CLASS_COLORS.size()];
        
        // 绘制边界框
        cv::rectangle(image, obj.bbox, box_color, 1);
        
        // 绘制点（每个点用不同颜色）
        for (size_t i = 0; i < obj.points.size(); i++) {
            cv::Scalar point_color = POINT_COLORS[i%POINT_COLORS.size()];
            cv::circle(image, obj.points[i], 5, point_color, -1);
            // 绘制点索引
            cv::putText(image, std::to_string(i+1), cv::Point(obj.points[i].x + 14, obj.points[i].y + 14), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, point_color, 2);
        }
        
        // 绘制类别和置信度
        std::string text = std::to_string(obj.label) + " " + std::to_string(obj.confidence).substr(0, 5);
        cv::putText(image, text, cv::Point(obj.bbox.x, obj.bbox.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2);
    }

    // 添加标题
    cv::putText(image, "Task: " + task_name, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
}

// 保存推理结果为JSON文件
void SaveResultsToJson(const std::vector<YoloObject>& objects, const std::string& img_path,
                       int img_width, int img_height, const std::string& output_path) {
    json root;
    root["imagePath"] = std::filesystem::path(img_path).filename().string();
    root["imageWidth"] = img_width;
    root["imageHeight"] = img_height;

    json objs = json::array();
    for (const auto& obj : objects) {
        json j;
        j["label"] = obj.label;
        j["confidence"] = obj.confidence;

        // bbox
        j["bbox"] = {{"x", obj.bbox.x}, {"y", obj.bbox.y},
                     {"width", obj.bbox.width}, {"height", obj.bbox.height}};

        // points
        json pts = json::array();
        for (const auto& p : obj.points) {
            pts.push_back({p.x, p.y});
        }
        j["points"] = pts;

        // point_confidences
        if (!obj.point_confidences.empty()) {
            json pcs = json::array();
            for (auto c : obj.point_confidences) pcs.push_back(c);
            j["point_confidences"] = pcs;
        }

        objs.push_back(j);
    }
    root["objects"] = objs;

    std::ofstream ofs(output_path);
    ofs << root.dump(2);
}

// 测试单个任务（is_segment=true 时使用掩码 IoU + 轮廓可视化）
bool TestTask(YoloTaskType task_type, const std::string& task_name,
              const std::string& model_path, const std::vector<std::string>& image_paths,
              const std::vector<std::string>& json_paths, const std::string& output_dir,
              bool is_segment = false) {

    YoloTRTInfer detector(task_type);
    
    if (!detector.LoadModel(model_path)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }
    std::cout << "Loaded model: " << model_path << std::endl;
    
    int total_images = 0;
    int total_detections = 0;
    int total_gt = 0;
    float total_error = 0.0f;
    int matched_count = 0;
    
    for (size_t i = 0; i < image_paths.size(); i++) {
        const std::string& img_path = image_paths[i];
        const std::string& json_path = json_paths[i];
        
        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << img_path << std::endl;
            continue;
        }
        
        int img_width, img_height;
        std::vector<Annotation> annotations = ReadAnnotations(json_path, img_width, img_height);
        if (annotations.empty()) {
            std::cerr << "Warning: No annotations found: " << json_path << std::endl;
        }
        
        total_images++;
        total_gt += annotations.size();
        
        auto objects = detector.Predict(image);
        total_detections += objects.size();
        
        // 保存推理结果JSON
        std::string json_out_path = output_dir + "/" + task_name + "_result_" + std::to_string(i) + ".json";
        SaveResultsToJson(objects, img_path, image.cols, image.rows, json_out_path);
        
        std::cout << "\n检测图像: " << std::filesystem::path(img_path).filename() << std::endl;
        std::cout << "  真值(GT)数:     " << annotations.size() << std::endl;
        std::cout << "  检测到目标数:    " << objects.size() << std::endl;
        
        // 按类别id匹配结果
        matched_count = 0;
        std::vector<bool> gt_matched_flag(annotations.size(), false);

        for (const auto& pred_obj : objects) {
            int pred_cls = pred_obj.label;
            int best_gt_idx = -1;
            float best_err = 0.0f;

            for (size_t j = 0; j < annotations.size(); ++j) {
                if (gt_matched_flag[j]) continue;

                int gt_cls = -1;
                auto it = label_map.find(annotations[j].label);
                if (it != label_map.end()) {
                    gt_cls = it->second;
                }

                if (gt_cls == pred_cls) {
                    if (is_segment) {
                        float iou = CalculateMaskIoU(pred_obj.points, annotations[j].points, img_width, img_height);
                        if (iou > best_err) {
                            best_err = iou;
                            best_gt_idx = static_cast<int>(j);
                        }
                    } else {
                        best_gt_idx = static_cast<int>(j);
                        break;
                    }
                }
            }

            if (best_gt_idx != -1) {
                gt_matched_flag[best_gt_idx] = true;
                matched_count++;

                if (is_segment) {
                    total_error += best_err;
                    std::cout << "  目标 ID = " << pred_cls
                              << " 置信度 = " << pred_obj.confidence
                              << " mIoU = " << best_err << " ✅" << std::endl;
                } else {
                    float err = CalculatePointError(pred_obj.points, annotations[best_gt_idx].points);
                    total_error += err;
                    std::cout << "  目标 ID = " << pred_cls
                            << " 置信度 = " << pred_obj.confidence
                            << " 误差 = " << err << " px ✅" << std::endl;
                }
            } else {
                std::cout << "  目标 ID = " << pred_cls << " 未匹配到真值 ❌" << std::endl;
            }
        }
        
        // 可视化
        if (is_segment) {
            for (const auto& obj : objects) {
                cv::Scalar color = CLASS_COLORS[obj.label % CLASS_COLORS.size()];
                if (obj.points.size() >= 3) {
                    std::vector<std::vector<cv::Point>> contours = {ptsToInt(obj.points)};
                    cv::drawContours(image, contours, -1, color, 2);
                    cv::Mat overlay = image.clone();
                    cv::fillPoly(overlay, contours, color);
                    cv::addWeighted(overlay, 0.3, image, 0.7, 0, image);
                }
                cv::rectangle(image, obj.bbox, color, 1);
                std::string text = std::to_string(obj.label) + " " + std::to_string(obj.confidence).substr(0, 5);
                cv::putText(image, text, cv::Point(obj.bbox.x, obj.bbox.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
            }
        } else {
            for (const auto& obj : objects) {
                cv::Scalar box_color = CLASS_COLORS[obj.label % CLASS_COLORS.size()];
                cv::rectangle(image, obj.bbox, box_color, 1);
                for (size_t k = 0; k < obj.points.size(); k++) {
                    cv::Scalar point_color = POINT_COLORS[k % POINT_COLORS.size()];
                    cv::circle(image, obj.points[k], 5, point_color, -1);
                    cv::putText(image, std::to_string(k+1), cv::Point(obj.points[k].x + 14, obj.points[k].y + 14),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, point_color, 2);
                }
                std::string text = std::to_string(obj.label) + " " + std::to_string(obj.confidence).substr(0, 5);
                cv::putText(image, text, cv::Point(obj.bbox.x, obj.bbox.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2);
            }
        }
        cv::putText(image, "Task: " + task_name, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        std::string output_path = output_dir + "/" + task_name + "_result_" + std::to_string(i) + ".jpg";
        cv::imwrite(output_path, image);
        std::cout << "  保存可视化结果到: " << output_path << std::endl;
    }
    
    std::cout << "\n========== " << task_name << " 误差统计结果 ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "图像总数:            " << total_images << "\n";
    std::cout << "GT对象总数:          " << total_gt << "\n";
    std::cout << "检测对象总数:        " << total_detections << "\n";
    std::cout << "匹配对象总数:        " << matched_count << "\n";
    std::cout << "精度:                " << (total_detections>0 ? (float)matched_count/total_detections : 0) << "\n";
    std::cout << "召回率:              " << (total_gt>0 ? (float)matched_count/total_gt : 0) << "\n";
    if (is_segment) {
        std::cout << "平均掩码IoU:         " << (matched_count>0 ? total_error/matched_count : 0) << "\n";
    } else {
        std::cout << "关键点误差平均距离:  " << (matched_count>0 ? total_error/matched_count : 0) << " px\n";
    }

    return true;
}

int main(int argc, char** argv) {
    // 数据和模型路径
    std::string data_root = "/root/perception/workspace/ebots_perception_core/vendor/CVAlgorithmRepo/src/data";
    if (argc > 1) {
        std::cout << "使用自定义数据路径: " << argv[1] << std::endl;
        data_root = argv[1];
    }
    std::string output_dir = data_root + "/infer_test/results";

    std::cout << "数据路径: " << data_root << std::endl;
    std::cout << "输出路径: " << output_dir << std::endl;
    
    // 创建输出目录
    std::filesystem::create_directories(output_dir);
    
    // 测试任务配置
    struct TaskConfig {
        YoloTaskType task_type;
        std::string task_name;
        std::string model_name;
        std::string data_dir;
        std::vector<std::string> image_extensions;
        bool is_segment = false;   // 是否为分割任务（使用不同的评估和可视化）
    };
    
    std::vector<TaskConfig> tasks = {
        {YoloTaskType::DETECT,  "detect",  "detect_yolo_detector.engine",  "detect",  {".png", ".tiff"}},
        {YoloTaskType::OBB,     "obb",     "obb_yolo_detector.engine",     "obb",     {".png", ".tiff"}},
        {YoloTaskType::POSE,    "pose",    "pose_yolo_detector.engine",    "pose",    {".png", ".tiff"}},
        {YoloTaskType::SEGMENT, "segment", "segment_yolo_detector.engine", "segment", {".png", ".tiff"}, true}
    };
    
    // 测试每个任务
    for (const auto& task : tasks) {
        std::string model_path = data_root + "/weights/" + task.model_name;
        std::string task_data_dir = data_root + "/infer_test/" + task.data_dir;

        std::cout << "\n====================== 测试任务 " << task.task_name << " ===================" << std::endl;
        std::cout << "  Model: " << model_path << std::endl;
        std::cout << "  Data: " << task_data_dir << std::endl;
        
        // 收集图片和JSON文件
        std::vector<std::string> image_paths;
        std::vector<std::string> json_paths;
        
        for (const auto& ext : task.image_extensions) {
            for (const auto& entry : std::filesystem::directory_iterator(task_data_dir)) {
                if (entry.path().extension() == ext) {
                    std::string base_name = entry.path().stem().string();
                    std::string json_path = task_data_dir + "/" + base_name + ".json";
                    
                    if (std::filesystem::exists(json_path)) {
                        image_paths.push_back(entry.path().string());
                        json_paths.push_back(json_path);
                    }
                }
            }
        }
        
        if (image_paths.empty()) {
            std::cerr << "No test data found for " << task.task_name << std::endl;
            continue;
        }

        TestTask(task.task_type, task.task_name, model_path, image_paths, json_paths, output_dir, task.is_segment);
    }
    
    std::cout << "\n============== 完成全部测试任务 ===============" << std::endl;
    return 0;
}