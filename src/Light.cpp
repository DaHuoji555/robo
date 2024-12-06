#include "../include/Light.h"

vector<vector<Point2f>> Light::light_match(vector<Light>& lights, Mat& frame) {
    vector<vector<Point2f>> quads; // 返回的所有四边形点矩阵集合

    if (lights.size() < 2) {
        cout << "Not enough lights to match." << endl;
        return quads; // 返回空
    }

    for (size_t i = 0; i < lights.size(); i++) {
        Light& light1 = lights[i];

        for (size_t j = i + 1; j < lights.size(); j++) {
            Light& light2 = lights[j];

            // 获取两个矩形的顶点
            Point2f vertices1[4], vertices2[4];
            light1.rect.points(vertices1);
            light2.rect.points(vertices2);

            // 按 x 坐标区分左右灯条
            Point2f rightPointsLeft[2], leftPointsRight[2];
            if (light1.rect.center.x < light2.rect.center.x) {
                // light1 在左，light2 在右
                sort(vertices1, vertices1 + 4, [](const Point2f& a, const Point2f& b) {
                    return a.x < b.x; // 按 x 坐标排序
                });
                sort(vertices2, vertices2 + 4, [](const Point2f& a, const Point2f& b) {
                    return a.x < b.x; // 按 x 坐标排序
                });

                rightPointsLeft[0] = vertices1[2]; // 左灯条的右侧两点
                rightPointsLeft[1] = vertices1[3];
                leftPointsRight[0] = vertices2[0]; // 右灯条的左侧两点
                leftPointsRight[1] = vertices2[1];
            } else {
                // light2 在左，light1 在右
                sort(vertices2, vertices2 + 4, [](const Point2f& a, const Point2f& b) {
                    return a.x < b.x; // 按 x 坐标排序
                });
                sort(vertices1, vertices1 + 4, [](const Point2f& a, const Point2f& b) {
                    return a.x < b.x; // 按 x 坐标排序
                });

                rightPointsLeft[0] = vertices2[2]; // 左灯条的右侧两点
                rightPointsLeft[1] = vertices2[3];
                leftPointsRight[0] = vertices1[0]; // 右灯条的左侧两点
                leftPointsRight[1] = vertices1[1];
            }

            // 计算 x 和 y 的差距
            if (abs(rightPointsLeft[0].x - leftPointsRight[0].x) > 300 ||
                abs(rightPointsLeft[0].y - leftPointsRight[0].y) > 300) {
                continue; // 跳过当前配对
            }

            // 计算中心点连线向量
            Point2f centerVector = light2.rect.center - light1.rect.center;

            // 计算左侧矩形向量
            Point2f leftVector = rightPointsLeft[1] - rightPointsLeft[0];

            // 计算右侧矩形向量
            Point2f rightVector = leftPointsRight[1] - leftPointsRight[0];

            // 计算向量积归一化值
            float crossLeft = abs(centerVector.x * leftVector.y - centerVector.y * leftVector.x) /
                              (sqrt(centerVector.x * centerVector.x + centerVector.y * centerVector.y) *
                               sqrt(leftVector.x * leftVector.x + leftVector.y * leftVector.y));

            float crossRight = abs(centerVector.x * rightVector.y - centerVector.y * rightVector.x) /
                               (sqrt(centerVector.x * centerVector.x + centerVector.y * centerVector.y) *
                                sqrt(rightVector.x * rightVector.x + rightVector.y * rightVector.y));

            // 判断条件：向量积绝对值差近似为 0
            if (abs(crossLeft - crossRight) < 0.1) { // 阈值可调

                // 上下延伸
                float h1 = light1.height; // 左灯条的高度
                float h2 = light2.height; // 右灯条的高度

                // 左灯条的延伸点
                Point2f extendedLeftTop = rightPointsLeft[0] + (rightPointsLeft[0] - rightPointsLeft[1]) * (h1 / light1.height);
                Point2f extendedLeftBottom = rightPointsLeft[1] + (rightPointsLeft[1] - rightPointsLeft[0]) * (h1 / light1.height);

                // 右灯条的延伸点
                Point2f extendedRightTop = leftPointsRight[0] + (leftPointsRight[0] - leftPointsRight[1]) * (h2 / light2.height);
                Point2f extendedRightBottom = leftPointsRight[1] + (leftPointsRight[1] - leftPointsRight[0]) * (h2 / light2.height);

                // 构造四边形的点
                vector<Point2f> quad = {extendedLeftTop, extendedLeftBottom, extendedRightBottom, extendedRightTop};

                // 将当前四边形点集加入结果集合
                quads.push_back(quad);
            }
        }
    }

    return quads;
}
