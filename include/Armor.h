#ifndef ARMOR_H
#define ARMOR_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

// 装甲板类
class Armor {
public:
    // 原始点和映射点
    Point2f originalPoints[4]; // 原始的四个点
    Point2f mappedPoints[4];   // 映射后的四个点（对应 200x200 矩阵）
    int number = 11;

    Armor(const vector<Point2f>& original);

    Mat transformToMatrix(const Mat& frame) const;


private:
    static vector<Point2f> sortPoints(const vector<Point2f>& points);
};



#endif //ARMOR_H
