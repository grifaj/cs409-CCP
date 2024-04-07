//
// Created by alfie on 04/04/2024.
//

#ifndef CPP_TEST_NCNN_H
#define CPP_TEST_NCNN_H

#include <opencv2/core.hpp>
#include <android/asset_manager.h>

/**
 * @brief Perform inference for classifying a given image using the NCNN model.
 * @param src: input image in OpenCV Mat format.
 * @param mgr: AAssetManager pointer for loading NCNN model files in assets folder.
 * @return the predicted class
 */
std::string Inference(cv::Mat& src, AAssetManager* mgr);
std::string Detection(cv::Mat& src, AAssetManager* mgr);

#endif //CPP_TEST_NCNN_H
