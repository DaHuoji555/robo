#include "../include/Judge_Light.h"

void filterAndAddLights(const vector<vector<Point>>& contours, vector<Light>& lights) {
    for (const auto& contour : contours) {
        RotatedRect rect = minAreaRect(contour);

        // 计算真实面积
        float realArea = contourArea(contour);

        // 跳过面积过小或过大的轮廓
        if (realArea < 150 || realArea > 2100) continue;

        // 获取宽和高
        float width = rect.size.width;
        float height = rect.size.height;

        // 确保长边是 height，短边是 width
        if (width > height) std::swap(width, height);

        // 跳过长宽比不符合条件的轮廓
        if (height / width < 2.0f) continue;

        // 获取旋转矩形的顶点
        Point2f vertices[4];
        rect.points(vertices);

        // 按 x 坐标排序
        std::sort(vertices, vertices + 4, [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
        });
        float avgX01 = (vertices[0].x + vertices[1].x) / 2.0f; // 索引 0, 1 的 x 坐标均值
        float avgX23 = (vertices[2].x + vertices[3].x) / 2.0f; // 索引 2, 3 的 x 坐标均值
        float m = abs(avgX01 - avgX23); // x 坐标的平均值差

        // 按 y 坐标排序
        std::sort(vertices, vertices + 4, [](const Point2f& a, const Point2f& b) {
            return a.y < b.y;
        });
        float avgY01 = (vertices[0].y + vertices[1].y) / 2.0f; // 索引 0, 1 的 y 坐标均值
        float avgY23 = (vertices[2].y + vertices[3].y) / 2.0f; // 索引 2, 3 的 y 坐标均值
        float n = abs(avgY01 - avgY23); // y 坐标的平均值差

        // 跳过 n/m 小于 2 的轮廓
        if (n / m < 2.0f) continue;

        // 保存灯条信息
        lights.emplace_back(realArea, rect.size.width * rect.size.height, width, height, rect.angle, rect);
    }
}

bool isRedOrYellowLight(const Mat& frame, const Light& light) {
    Mat hsv, mask, innerMask;

    // 转换为 HSV
    cvtColor(frame, hsv, COLOR_BGR2HSV);

    // 创建灯条的掩码
    mask = Mat::zeros(frame.size(), CV_8UC1);
    Point2f vertices[4];
    light.rect.points(vertices);
    vector<Point> contour(vertices, vertices + 4);
    fillConvexPoly(mask, contour, Scalar(255));

    // 生成内层掩码（腐蚀后的掩码）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5)); // 调整内层范围
    erode(mask, innerMask, kernel);

    // 计算边缘掩码：外层减去内层
    Mat edgeMask = mask - innerMask;

    // 转换为 HSV 范围进行红色检测
    Mat redMask1, redMask2, combinedRedMask;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), redMask1);  // 红色范围1
    inRange(hsv, Scalar(170, 100, 100), Scalar(180, 255, 255), redMask2); // 红色范围2

    // 合并红色掩码
    combinedRedMask = redMask1 | redMask2;

    // 获取边缘红色部分
    Mat edgeRedMask = combinedRedMask & edgeMask;

    // 判断红色像素数量是否足够
    double validPixelCount = countNonZero(edgeRedMask);

    // 判断条件：红色边缘像素占整个灯条面积的一定比例
    return validPixelCount > 0.05 * countNonZero(mask); // 5% 的边缘是红色
}
