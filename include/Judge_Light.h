#ifndef JUDGE_LIGHT_H
#define JUDGE_LIGHT_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/ml.hpp>
#include "../include/Armor.h"
#include "../include/Light.h"
#include "../include/KNN.h"


using namespace cv;
using namespace std;
using namespace cv::ml;


void filterAndAddLights(const vector<vector<Point>>& contours, vector<Light>& lights);

bool isRedOrYellowLight(const Mat& frame, const Light& light);


#endif //JUDGE_LIGHT_H
