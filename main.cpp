#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/ml.hpp>
#include "include/Armor.h"
#include "include/Light.h"
#include "include/KNN.h"
#include "include/Judge_Light.h"

using namespace cv;
using namespace std;
using namespace cv::ml;



int main() {

    Ptr<KNearest> knn = trainKNNFromDataset("../dataset");

    // 打开视频文件
    string videoPath = "test3.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Error: Unable to open video file." << endl;
        return -1;
    }

    Mat frame, gray, binary, morph;
    namedWindow("Original Frame", WINDOW_NORMAL);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 转为灰度图
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 二值化
        threshold(gray, binary, 220, 255, THRESH_BINARY);



        // 创建内核
        Mat erosionKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat dilationKernel = getStructuringElement(MORPH_RECT, Size(5, 5));

        // 临时变量用于存储中间结果
        Mat temp;

        // 腐蚀操作（2 次）
        erode(binary, temp, erosionKernel, Point(-1, -1), 2);

        // 膨胀操作（3 次）
        dilate(temp, morph, dilationKernel, Point(-1, -1), 2);

        // 展示结果（可选）
        imshow("Binary after Morphology", morph);


        // 查找轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(morph, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 过滤和保存灯条
        vector<Light> lights;
        filterAndAddLights(contours, lights);

        // //红色匹配
        //  for (int i = 0; i < lights.size();) {
        //      if (!isRedOrYellowLight(frame, lights[i])) {
        //          lights.erase(lights.begin() + i); // 删除当前不符合条件的灯条
        //      } else {
        //          i++; // 如果符合条件，继续检查下一个
        //      }
        //  }

        vector<vector<Point2f>> boards;
        vector<Armor> armors;

        // 匹配灯条成对
        boards = Light::light_match(lights, frame);

        for(auto & board : boards) {
            armors.emplace_back(board);
        }

        for (auto it = armors.begin(); it != armors.end(); ) {
            it->number = predictLabel(knn, it->transformToMatrix(frame)); // 预测标签
            if (it -> number == 10) {
                it = armors.erase(it); // 如果标签为 10，删除当前装甲板
            } else {
                ++it; // 否则递增迭代器
            }
        }


        // 在原图像上绘制灯条
        for (const auto& light : lights) {
                // 绘制外接矩形
                Point2f vertices[4];
                light.rect.points(vertices);
                for (int j = 0; j < 4; j++) {
                    line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2); // 绿色边框
                }

                // 显示灯条的外接矩形面积和真实面积
                string areaText = format("Real: %.1f, Box: %.1f", light.realArea, light.boundingArea);
                putText(frame, areaText, light.rect.center + Point2f(0, -10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);

                // 显示灯条的旋转角度
                string angleText = format("Angle: %.1f", light.angle);
                putText(frame, angleText, light.rect.center + Point2f(0, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
        }

        // 遍历装甲板数组并绘制四边形和中心点
        for (const auto& armor : armors) {
            // 绘制装甲板的四边形
            for (int k = 0; k < 4; k++) {
                line(frame, armor.originalPoints[k], armor.originalPoints[(k + 1) % 4], Scalar(0, 255, 0), 2); // 绿色四边形
            }

            // 计算装甲板的中心点
            Point2f center = (armor.originalPoints[0] + armor.originalPoints[2]) * 0.5; // 使用对角点计算中心点
            circle(frame, center, 5, Scalar(0, 0, 255), -1); // 红色圆点表示中心

            // 显示中心点坐标
            string centerText = format("(%d, %d)", (int)center.x, (int)center.y);
            putText(frame, centerText, center + Point2f(10, 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);

            putText(frame, format("(%d)", armor.number),
                    center + Point2f(20, 20),
                    FONT_HERSHEY_SIMPLEX,
                    1.0,
                    Scalar(255, 0, 255),
                    2);

        }


        // 显示原始帧
        imshow("Original Frame", frame);

        // 按 'q' 键退出
        char c = (char)waitKey(1);
        if (c == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
