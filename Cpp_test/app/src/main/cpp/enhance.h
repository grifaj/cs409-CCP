//
// Created by alfie on 26/01/2024.
//

#ifndef CPP_TEST_ENHANCE_H
#define CPP_TEST_ENHANCE_H

#include "opencv2/imgproc.hpp"

cv::Mat captureImage(AAssetManager* mgr, cv::Mat srcImg);
void loadTranslationModel();

#endif //CPP_TEST_ENHANCE_H
