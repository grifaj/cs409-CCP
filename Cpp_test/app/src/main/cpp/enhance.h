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
cv::Mat grayImage(cv::Mat* srcImg);
cv::Mat binariseBox(cv::Mat img, cv::Rect inBox);
void sortParallelVector(std::vector<cv::Rect>* vec, std::vector<float>* score_vec);
void nms(std::vector<cv::Rect>* boxes, std::vector<float>* scores, std::vector<cv::Rect>* selected, float thresh);
cv::Mat padImage(cv::Mat *srcImg);
cv::Mat resizeSF(cv::Mat *srcImg, float* sf);

#endif //CPP_TEST_ENHANCE_H
