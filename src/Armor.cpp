#include "../include/Armor.h"

// 构造函数：接收 vector<Point2f> 数组
   Armor::Armor(const vector<Point2f>& original) {
       if (original.size() != 4) {
           throw invalid_argument("Original points size must be 4.");
       }

       // 对原始点进行排序
       vector<Point2f> sortedPoints = sortPoints(original);

       // 保存排序后的点
       for (int i = 0; i < 4; i++) {
           originalPoints[i] = sortedPoints[i];
       }

       // 设置映射点
       mappedPoints[0] = Point2f(0, 0);           // 左上角
       mappedPoints[1] = Point2f(0, 200);         // 左下角
       mappedPoints[2] = Point2f(200, 200);       // 右下角
       mappedPoints[3] = Point2f(200, 0);         // 右上角
   }

// 透射变换，将原始图像变换成 200x200 的矩阵
Mat Armor::transformToMatrix(const Mat& frame) const {
       // 深拷贝原图像
       Mat imageCopy = frame.clone();
       Mat grayImage;
       cvtColor(imageCopy, grayImage, COLOR_BGR2GRAY);

       // // 如果像素值太大（白），直接变成 0
       // for (int i = 0; i < grayImage.rows; ++i) {
       //     for (int j = 0; j < grayImage.cols; ++j) {
       //         if (grayImage.at<uchar>(i, j) > 50) {
       //             grayImage.at<uchar>(i, j) = 0;
       //         }
       //     }
       // }

       //二值化test1，3
       Mat binaryImage;
       threshold(grayImage, binaryImage, 20, 255, THRESH_BINARY);

       // //二值化test2
       //  Mat binaryImage;
       //  threshold(grayImage, binaryImage, 100, 255, THRESH_BINARY);

       // 腐蚀操作，使用矩形核使白色变薄
       Mat erodedImage;
       Mat erosionKernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 使用 3x3 的矩形核
       erode(binaryImage, erodedImage, erosionKernel);
       erode(erodedImage, erodedImage, erosionKernel);

       imageCopy = erodedImage.clone();

       // 计算透射变换矩阵
       Mat transformMatrix = getPerspectiveTransform(originalPoints, mappedPoints);

       // 执行透射变换
       Mat result;
       warpPerspective(imageCopy, result, transformMatrix, Size(200, 200));
       imshow("Processed Result", result); // 显示结果

       return result;

   }


// 对点进行排序，确保顺序为：左上、左下、右下、右上
vector<Point2f> Armor::sortPoints(const vector<Point2f>& points) {
       if (points.size() != 4) {
           throw invalid_argument("Points size must be 4.");
       }

       // 先按 X 坐标排序，找到左右两侧的点
       vector<Point2f> sortedPoints = points;
       sort(sortedPoints.begin(), sortedPoints.end(), [](const Point2f& a, const Point2f& b) {
           return a.x < b.x;
       });

       // 左侧的两个点
       vector<Point2f> leftPoints = {sortedPoints[0], sortedPoints[1]};
       // 右侧的两个点
       vector<Point2f> rightPoints = {sortedPoints[2], sortedPoints[3]};

       // 按 Y 坐标进一步排序，区分上下
       sort(leftPoints.begin(), leftPoints.end(), [](const Point2f& a, const Point2f& b) {
           return a.y < b.y;
       });
       sort(rightPoints.begin(), rightPoints.end(), [](const Point2f& a, const Point2f& b) {
           return a.y < b.y;
       });

       // 返回按顺序排列的点：左上、左下、右下、右上
       return {leftPoints[0], leftPoints[1], rightPoints[1], rightPoints[0]};
   }