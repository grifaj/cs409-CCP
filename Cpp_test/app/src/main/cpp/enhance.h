//
// Created by alfie on 26/01/2024.
//

#ifndef CPP_TEST_ENHANCE_H
#define CPP_TEST_ENHANCE_H

#include "opencv2/imgproc.hpp"

cv::Mat captureImage(AAssetManager* mgr, cv::Mat srcImg, int option);
cv::Mat captureBoxImage(AAssetManager* mgr, cv::Mat srcImg, int x, int y, int w, int h);
void preloadModels(AAssetManager* mgr);
float calculate_IOU(cv::Rect a, cv::Rect b);

#endif //CPP_TEST_ENHANCE_H
