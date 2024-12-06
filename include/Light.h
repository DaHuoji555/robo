#ifndef LIGHT_H
#define LIGHT_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

// 灯条类
class Light {
public:
    float realArea;         // 真实面积（轮廓面积）
    float boundingArea;     // 外接矩形面积
    float width;            // 外接矩形宽度
    float height;           // 外接矩形高度
    float angle;            // 外接矩形的旋转角度
    RotatedRect rect;       // 外接矩形

    Light(float realArea, float boundingArea, float width, float height, float angle, RotatedRect rect)
        : realArea(realArea), boundingArea(boundingArea), width(width), height(height), angle(angle), rect(rect) {}

    static vector<vector<Point2f>> light_match(vector<Light>& lights, Mat& frame);

};

#endif //LIGHT_H
