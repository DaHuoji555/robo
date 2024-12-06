#ifndef KNN_H
#define KNN_H

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <filesystem>
#include <vector>
#include <string>

using namespace cv;
using namespace std;
using namespace cv::ml;
namespace fs = std::filesystem;

void loadDataset(const string& datasetPath, Mat& features, Mat& labels);

Ptr<KNearest> trainKNNFromDataset(const string& datasetPath);

int predictLabel(const Ptr<KNearest>& knn, const Mat& inputBlock);

#endif //KNN_H
