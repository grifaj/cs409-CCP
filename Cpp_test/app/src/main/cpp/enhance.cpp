#include <vector>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <android/asset_manager.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include "enhance.h"
#include <string>

#include "net.h"
#include "platform.h"

ncnn::Net translationModel;
ncnn::Net detectionModel;
bool modelInitialisedFlag = false;
bool detmodelInitialisedFlag = false;
AAssetManager* mgr;

void loadTranslationModel() {
    // Load model
    int ret = translationModel.load_param(mgr,"mobilenet_v3_large_3-sim-opt.param");
    if (ret) {
         __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "mobilenet_v3_large_3-sim-opt.bin");
    if (ret) {
       __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    modelInitialisedFlag = true;
}

void preloadModels(AAssetManager* manager) {
    auto beg = std::chrono::high_resolution_clock::now();
    mgr = manager;

    int ret = translationModel.load_param(mgr,"mobilenet_v3_large-sim-opt.param");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "mobilenet_v3_large-sim-opt.bin");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    modelInitialisedFlag = true;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "load translate model %f", duration.count()/1000.0);

    ret = detectionModel.load_param(mgr,"model.ncnn.param");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = detectionModel.load_model(mgr, "model.ncnn.bin");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    detmodelInitialisedFlag = true;
}

cv::Mat binariseBox(cv::Mat img, cv::Rect inBox)
{
    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    cv::Mat boxImg(grayImg, inBox);

    //smooth the box - possibly remove?
    cv::Mat boxSmoothed;
    cv::medianBlur(boxImg, boxSmoothed, 5);

    cv::Mat threshBox;

    cv::threshold(boxSmoothed, threshBox, 0, 255, cv::THRESH_OTSU);
    boxImg.release();
    boxSmoothed.release();
    grayImg.release();

    int rows = threshBox.rows;
    int cols = threshBox.cols;

    int box_tl = (int) threshBox.at<uchar>(0, 0);
    int box_bl = (int) threshBox.at<uchar>(rows - 1, 0);
    int box_tr = (int) threshBox.at<uchar>(0, cols-1);
    int box_br = (int) threshBox.at<uchar>(rows - 1, cols - 1);

    int sum_corners = box_tl + box_bl + box_tr + box_br;

    if (sum_corners <= 510)
    {
        cv::bitwise_not(threshBox, threshBox);
    }

    cv::Mat threshBGR;
    cvtColor(threshBox, threshBGR, cv::COLOR_GRAY2BGR);

    return threshBGR;
}


void displayOverlay(cv::Mat colImg, cv::Rect location, cv::Mat replaceImg, int option){

    std::string bugString;

    bugString = "Starting box extraction";
    __android_log_print(ANDROID_LOG_DEBUG, "binary box", "%s", bugString.c_str());

    if(!modelInitialisedFlag)
    {
        loadTranslationModel();
    }

    int x_check = location.x < colImg.cols && location.x > 0;
    int y_check = location.y < colImg.rows && location.y > 0;
    int w_check = location.x+location.width < colImg.cols && location.x+location.width > 0;
    int h_check = location.y+location.height < colImg.rows && location.y+location.height > 0;

    int total_check = x_check && y_check && w_check && h_check;

    // get input from bounding box
    if (total_check)
    {
        cv::Rect roiRect(location);
        cv::Mat roi = colImg(roiRect);
        cv::Mat replaceroi = replaceImg(roiRect);

        /*
        * Test that image is correct extracted dimensions from rectangle.
         * Test that values are equal in the two regions
        */

        // binarise image
        cv::Mat binRoi = binariseBox(colImg, roiRect);

        bugString = "Retrieved binary box";
        __android_log_print(ANDROID_LOG_DEBUG, "binary box", "%s", bugString.c_str());

        cv::Mat binRoiR;
        cv::resize(binRoi, binRoiR, cv::Size(232, 232));

        /*
        * Test that image is resized to 232 x 232
        */

        const int cropSize = 224;
        const int offsetW = (binRoiR.cols - cropSize) / 2;
        const int offsetH = (binRoiR.rows - cropSize) / 2;
        const cv::Rect roiBin(offsetW, offsetH, cropSize, cropSize);
        binRoi = binRoiR(roiBin).clone();

        /*
        * Test that image is resized to 224 x 224
        */

        bugString = "Processed binary box";
        __android_log_print(ANDROID_LOG_DEBUG, "binary box", "%s", bugString.c_str());

        binRoiR.release();

        // Convert image data to ncnn format
        // opencv image in bgr, model needs rgb
        ncnn::Mat input = ncnn::Mat::from_pixels(binRoi.data, ncnn::Mat::PIXEL_BGR2RGB, binRoi.cols,
                                                 binRoi.rows);

        float means[] = {0.485f*255.f, 0.456f*255.f, 0.406*255.f};
        float norms[] = {1/0.229f/255.f, 1/0.224/255.f, 1/0.225f/255.f};
        input.substract_mean_normalize(means, norms);

        // Inference
        ncnn::Extractor extractor = translationModel.create_extractor();
        //extractor.set_light_mode(true);
        extractor.input("input", input);
        ncnn::Mat output;
        extractor.extract("output", output);

        float max = 0.0;
        std::string argMax;
        for (int j = 0; j < output.w; j++) {
            if (output[j] > max) {
                max = output[j];
                argMax = std::to_string(j+1);

                bugString = "Class: " + std::to_string(j);
                __android_log_print(ANDROID_LOG_DEBUG, "translation model", "%s", bugString.c_str());
            }
        }

        /*
        * check that confidence is <= 1 and that max class is between 1 and 1000
        */

        bugString = "Max class: " + argMax;
        __android_log_print(ANDROID_LOG_DEBUG, "translation model", "%s", bugString.c_str());

        bugString = "Max confidence: " + std::to_string(max);
        __android_log_print(ANDROID_LOG_DEBUG, "translation model", "%s", bugString.c_str());

        // get file name
        float confThreshold = 0.7;
        if (max >= confThreshold)
        {

            /*
            * Check confidence threshold is always > 0.7
            */

            std::string filename = "overlays/";
            filename.append(argMax);
            filename.append(".bmp");

            // load file from assets
            AAsset *asset = AAssetManager_open(mgr, filename.c_str(), 0);
            long size = AAsset_getLength(asset);
            uchar *buffer = (uchar *) malloc(sizeof(uchar) * size);
            AAsset_read(asset, buffer, size);
            AAsset_close(asset);

            // convert file to rgb image
            cv::Mat rawData(1, size, CV_8UC1, (void *) buffer);
            cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);

            cv::Mat decodeColor;
            cvtColor(decodedImage, decodeColor, cv::COLOR_BGR2RGB);
            decodedImage.release();

            //overlay image on rectangle

            cv::Mat overlayImg;
            cv::resize(decodeColor, overlayImg, roi.size());
            decodeColor.release();

            cv::Mat alphaMask(roi.rows, roi.cols, CV_8UC1, cv::Scalar(255));

            if (option == 1)
            {
                cv::cvtColor(overlayImg, overlayImg, cv::COLOR_RGB2RGBA);
                cv::cvtColor(roi, roi, cv::COLOR_BGR2RGBA);

                std::vector<cv::Mat>channels(4);
                cv::split(overlayImg, channels);

                channels[3] = alphaMask;

                cv::merge(channels, overlayImg);
            }

            addWeighted(overlayImg, 1, roi, 0, 0, replaceroi);
            overlayImg.release();

            /*
            * Check correct overlay.
            */

        }
    }
}

//cv::Mat Detection(cv::Mat src, cv::Mat orig, int option) {
//    auto beg = std::chrono::high_resolution_clock::now();
//
//    std::string bugString;
//
//    cv::Mat srcScaled;
//    float sf;
//    bool r_or_c;
//
//    if(src.rows >= src.cols)
//    {
//        sf = 512.0/(float)src.rows;
//        r_or_c = false;
//    }
//    else
//    {
//        sf = 512.0/(float)src.cols;
//        r_or_c = true;
//    }
//
//    cv::resize(src, srcScaled, cv::Size(), sf, sf);
//
//    /*
//     * Check that image has been resized with one dimension = 512 and the other scaled by the same s.f.
//     */
//
//    cv::Mat srcFinal;
//
//    if (r_or_c)
//    {
//        cv::copyMakeBorder(srcScaled, srcFinal, 0, 512-srcScaled.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
//    }
//    else
//    {
//        cv::copyMakeBorder(srcScaled, srcFinal, 0, 0, 0, 512-srcScaled.cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
//    }
//
//    /*
//     * Check that image now has 512 x 512 dimensions after padding
//     */
//
//    // Convert image data to ncnn format
//    // opencv image in bgr, model needs rgb
//    if(!detmodelInitialisedFlag){
//        loadDetectionModel();
//    }
//
//    bugString = "Height: " + std::to_string(srcFinal.rows);
//    __android_log_print(ANDROID_LOG_DEBUG, "resize", "%s", bugString.c_str());
//    bugString = "Width: " + std::to_string(srcFinal.cols);
//    __android_log_print(ANDROID_LOG_DEBUG, "resize", "%s", bugString.c_str());
//
//    ncnn::Mat input = ncnn::Mat::from_pixels(srcFinal.data,
//                                             ncnn::Mat::PixelType::PIXEL_RGB,
//                                             srcFinal.cols, srcFinal.rows);
//
//
//    float means[] = {0.0, 0.0, 0.0};
//    float norms[] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
//    input.substract_mean_normalize(means, norms);
//
//    // Inference
//    ncnn::Extractor extractor = detectionModel.create_extractor();
//    extractor.input("in0", input);
//    ncnn::Mat output;
//    extractor.extract("out0", output);
//
//    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);
//    std::vector<cv::Rect>* boxes = new std::vector<cv::Rect>();
//    std::vector<float>* confidences = new std::vector<float>();
//
//    float confidenceThresh = 0.65;
//    int sec_size = out_flatterned.w/5;
//    for (int j=0; j<sec_size; j++)
//    {
//        if (out_flatterned[j+(sec_size*4)] > confidenceThresh)
//        {
//            float x = out_flatterned[j] / sf;
//            float y = out_flatterned[j+(sec_size)] / sf;
//            float w = out_flatterned[j+(sec_size*2)] / sf;
//            float h = out_flatterned[j+(sec_size*3)] / sf;
//
//            int left = int((x - 0.5 * w));
//            int top = int((y - 0.5 * h));
//
//            if (left > 0 && top > 0 && w > 0 && h > 0 &&
//                left+w < orig.cols && top+h < orig.rows)
//            {
//                int width = int(w);
//                int height = int(h);
//                boxes->push_back(cv::Rect(left, top, width, height));
//                confidences->push_back(float(out_flatterned[j+(sec_size*4)]));
//            }
//        }
//    }
//
//    /*
//     * Check that all boxes given are within image co-ordinates and all confidences are above confidence thresh
//     */
//
//    bugString = "Detected " + std::to_string(boxes->size()) + " boxes";
//    __android_log_print(ANDROID_LOG_DEBUG, "det_boxes", "%s", bugString.c_str());
//
//    std::vector<cv::Rect>* selected_boxes = new std::vector<cv::Rect>();
//    nms(boxes, confidences, selected_boxes, 0.5);
//
//    bugString = "NMS " + std::to_string(selected_boxes->size()) + " boxes";
//    __android_log_print(ANDROID_LOG_DEBUG, "det_boxes", "%s", bugString.c_str());
//
//    auto end = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
//    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "yolo dectection %f", duration.count()/1000.0);
//
//    beg = std::chrono::high_resolution_clock::now();
//
//    cv::Mat overlay(orig.rows, orig.cols, CV_8UC4, cv::Scalar(0,0,0,0));
//    for (std::size_t i = 0; i != selected_boxes->size(); i++)
//    {
//        //cv::rectangle(src, (*selected_boxes)[i], cv::Scalar(255, 0, 0), 1);
//        if (option == 1)
//        {
//            displayOverlay(orig, (*selected_boxes)[i], overlay, option);
//        }
//        else
//        {
//            displayOverlay(orig, (*selected_boxes)[i], orig, option);
//        }
//
//    }
//    end = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
//    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "translation inference %f with %d boxes", duration.count()/1000.0, (int) selected_boxes->size());
//
//    if (option == 1)
//    {
//        return overlay;
//    }
//    else
//    {
//        return orig;
//    }
//}

cv::Mat grayImage(cv::Mat* srcImg)
{
    cv::Mat img;
    cvtColor(*srcImg, img, cv::COLOR_RGBA2BGR);

    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    return grayImg;
}

cv::Mat preProcessImage(cv::Mat* srcImg)
{
    cv::Mat grayDilate;
    cv::Mat grayErode;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(*srcImg, grayDilate, kernel, cv::Point(-1, -1), 1);
    cv::erode(grayDilate, grayErode, kernel, cv::Point(-1, -1), 1);


    cv::Mat graySmoothed;
    cv::medianBlur(grayErode, graySmoothed, 5);

    grayDilate.release();
    grayErode.release();

    return graySmoothed;
}

cv::Mat grayToBGR(cv::Mat* srcImg)
{
    cv::Mat grayBGR;
    cvtColor(*srcImg, grayBGR, cv::COLOR_GRAY2BGR);

    return grayBGR;
}

cv::Mat resizeSF(cv::Mat *srcImg, float* sf)
{
    cv::Mat srcScaled;

    if(srcImg->rows >= srcImg->cols)
    {
        *sf = 512.0/(float)srcImg->rows;
    }
    else
    {
        *sf = 512.0/(float)srcImg->cols;
    }

    cv::resize(*srcImg, srcScaled, cv::Size(), *sf, *sf);

    return srcScaled;
}

cv::Mat padImage(cv::Mat *srcImg)
{
    cv::Mat srcFinal;
    bool r_or_c;

    if (srcImg->rows >= srcImg->cols)
    {
        r_or_c = false;
    }
    else
    {
        r_or_c = true;
    }

    if (r_or_c)
    {
        cv::copyMakeBorder(*srcImg, srcFinal, 0, 512-srcImg->rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    else
    {
        cv::copyMakeBorder(*srcImg, srcFinal, 0, 0, 0, 512-srcImg->cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }

    return srcFinal;
}

void detectModel(cv::Mat *srcImg, cv::Mat *orig, float* sf, std::vector<cv::Rect>* boxes, std::vector<float>* confidences)
{
    ncnn::Mat input = ncnn::Mat::from_pixels(srcImg->data,
                                             ncnn::Mat::PixelType::PIXEL_RGB,
                                             srcImg->cols, srcImg->rows);


    float means[] = {0.0, 0.0, 0.0};
    float norms[] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    input.substract_mean_normalize(means, norms);

    // Inference
    ncnn::Extractor extractor = detectionModel.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat output;
    extractor.extract("out0", output);

    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);

    float confidenceThresh = 0.65;
    int sec_size = out_flatterned.w/5;
    for (int j=0; j<sec_size; j++)
    {
        if (out_flatterned[j+(sec_size*4)] > confidenceThresh)
        {
            float x = out_flatterned[j] / *sf;
            float y = out_flatterned[j+(sec_size)] / *sf;
            float w = out_flatterned[j+(sec_size*2)] / *sf;
            float h = out_flatterned[j+(sec_size*3)] / *sf;

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            if (left > 0 && top > 0 && w > 0 && h > 0 &&
                left+w < orig->cols && top+h < orig->rows)
            {
                int width = int(w);
                int height = int(h);
                boxes->push_back(cv::Rect(left, top, width, height));
                confidences->push_back(float(out_flatterned[j+(sec_size*4)]));
            }
        }
    }
}

void loadDetectionModel()
{
    int ret = detectionModel.load_param(mgr,"model.ncnn.param");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = detectionModel.load_model(mgr, "model.ncnn.bin");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    detmodelInitialisedFlag = true;
}

void sortParallelVector(std::vector<cv::Rect>* vec, std::vector<float>* score_vec)
{

    std::vector<cv::Rect>& vec_ref = *vec;
    std::vector<float>& score_ref = *score_vec;

    std::vector<std::size_t> index_vec;
    std::vector<cv::Rect> vec_ordered;
    std::vector<float> score_vec_ordered;

    for (std::size_t i = 0; i != vec->size(); ++i)
    {
        index_vec.push_back(i);
    }

    std::sort(
            index_vec.begin(), index_vec.end(),
            [&](std::size_t a, std::size_t b) {return score_ref[a] > score_ref[b];});

    for (std::size_t i = 0; i != index_vec.size(); ++i)
    {
        vec_ordered.push_back(vec_ref[index_vec[i]]);
        score_vec_ordered.push_back(score_ref[index_vec[i]]);
    }

    *vec = vec_ordered;
    *score_vec = score_vec_ordered;

}

float calculate_IOU(cv::Rect a, cv::Rect b)
{
    float areaA = a.area();

    float areaA_br_x = a.br().x;
    float areaA_br_y = a.br().y;

    if (areaA <= 0.0)
    {
        return 0.0;
    }

    float areaB = b.area();

    if (areaB <= 0.0)
    {
        return 0.0;
    }
    float areaB_br_x = b.br().x;
    float areaB_br_y = b.br().y;

    float intersection_left_x = std::max(a.tl().x, b.tl().x);
    float intersection_left_y = std::max(a.tl().y, b.tl().y);
    float intersection_bottom_x = std::min(areaA_br_x, areaB_br_x);
    float intersection_bottom_y = std::min(areaA_br_y, areaB_br_y);

    float intersection_width = std::max(intersection_bottom_x - intersection_left_x, (float)0.0);
    float intersection_height = std::max(intersection_bottom_y - intersection_left_y, (float)0.0);

    float intersection_area = intersection_width * intersection_height;

    return (float) intersection_area / (float) (areaA + areaB - intersection_area);

}

void nms(std::vector<cv::Rect>* boxes, std::vector<float>* scores, std::vector<cv::Rect>* selected, float thresh)
{

    sortParallelVector(boxes, scores);

    std::vector<cv::Rect>& boxes_ref = *boxes;
    //std::vector<float>& score_ref = *scores;

    std::vector<bool> active;

    for (std::size_t i = 0; i != boxes->size(); i++)
    {
        active.push_back(true);
    }

    int num_active = active.size();

    bool done = false;

    for (std::size_t i = 0; i != boxes->size(); i++)
    {
        if (active[i])
        {
            cv::Rect box_a = boxes_ref[i];
            selected->push_back(box_a);

            for (std::size_t j = i+1; j != boxes->size(); j++)
            {
                if (active[j])
                {
                    cv::Rect box_b = boxes_ref[j];

                    float iou = calculate_IOU(box_a, box_b);

                    if (iou > thresh)
                    {
                        active[j] = false;
                        num_active--;

                        if (num_active <= 0)
                        {
                            done = true;
                            break;
                        }
                    }
                }
            }

            if (done)
            {
                break;
            }
        }
    }
}

cv::Mat translationPreProcess(cv::Mat* srcImg)
{
    cv::Mat binRoiR;
    cv::resize(*srcImg, binRoiR, cv::Size(232, 232));

    const int cropSize = 224;
    const int offsetW = (binRoiR.cols - cropSize) / 2;
    const int offsetH = (binRoiR.rows - cropSize) / 2;
    const cv::Rect roiBin(offsetW, offsetH, cropSize, cropSize);
    return binRoiR(roiBin);
}

void getTranslation(cv::Mat* srcImg, float* max, std::string* argMax)
{
    ncnn::Mat input = ncnn::Mat::from_pixels(srcImg->data, ncnn::Mat::PIXEL_BGR2RGB, srcImg->cols,
                                             srcImg->rows);

    float means[] = {0.485f*255.f, 0.456f*255.f, 0.406*255.f};
    float norms[] = {1/0.229f/255.f, 1/0.224/255.f, 1/0.225f/255.f};
    input.substract_mean_normalize(means, norms);

    // Inference
    ncnn::Extractor extractor = translationModel.create_extractor();
    //extractor.set_light_mode(true);
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);

    *max = 0.0;
    for (int j = 0; j < output.w; j++) {
        if (output[j] > *max) {
            *max = output[j];
            *argMax = std::to_string(j+1);
        }
    }
}

void overlayTranslation(cv::Mat roi, cv::Mat replaceroi, float* max, std::string* argMax, int option)
{
    float confThreshold = 0.7;
    if (*max >= confThreshold)
    {
        std::string filename = "overlays/";
        filename.append(*argMax);
        filename.append(".bmp");

        // load file from assets
        AAsset *asset = AAssetManager_open(mgr, filename.c_str(), 0);
        long size = AAsset_getLength(asset);
        uchar *buffer = (uchar *) malloc(sizeof(uchar) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);

        // convert file to rgb image
        cv::Mat rawData(1, size, CV_8UC1, (void *) buffer);
        cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);

        cv::Mat decodeColor;
        cvtColor(decodedImage, decodeColor, cv::COLOR_BGR2RGB);
        decodedImage.release();

        //overlay image on rectangle

        cv::Mat overlayImg;
        cv::resize(decodeColor, overlayImg, roi.size());
        decodeColor.release();

        cv::Mat alphaMask(roi.rows, roi.cols, CV_8UC1, cv::Scalar(255));

        cv::cvtColor(overlayImg, overlayImg, cv::COLOR_RGB2RGBA);
        cv::cvtColor(roi, roi, cv::COLOR_BGR2RGBA);

        std::vector<cv::Mat>channels(4);
        cv::split(overlayImg, channels);

        channels[3] = alphaMask;

        cv::merge(channels, overlayImg);

        addWeighted(overlayImg, 1, roi, 0, 0, replaceroi);

        overlayImg.release();
    }
}

cv::Mat captureImage(AAssetManager* manager, cv::Mat srcImg, int option) {
    auto beg = std::chrono::high_resolution_clock::now();
    mgr = manager;

    cv::Mat grayImg = grayImage(&srcImg);

    cv::Mat graySmoothed = preProcessImage(&grayImg);
    grayImg.release();

    cv::Mat grayBGR = grayToBGR(&graySmoothed);
    graySmoothed.release();

    float sf;
    cv::Mat sfScaled = resizeSF(&grayBGR, &sf);
    grayBGR.release();

    cv::Mat imPad = padImage(&sfScaled);
    sfScaled.release();

    if(!detmodelInitialisedFlag)
    {
        loadDetectionModel();
    }

    std::vector<cv::Rect>* boxes = new std::vector<cv::Rect>();
    std::vector<float>* confidences = new std::vector<float>();
    detectModel(&imPad, &srcImg, &sf, boxes, confidences);

    std::vector<cv::Rect>* selected_boxes = new std::vector<cv::Rect>();
    nms(boxes, confidences, selected_boxes, 0.5);

    if(!modelInitialisedFlag)
    {
        loadTranslationModel();
    }

    cv::Mat overlay(srcImg.rows, srcImg.cols, CV_8UC4, cv::Scalar(0,0,0,0));
    cv::Mat replaceImg;
    if (option == 1)
    {
        replaceImg = overlay;
    }
    else
    {
        replaceImg = srcImg;
    }

    float max_class;
    std::string argMax_class;
    for (std::size_t i = 0; i != selected_boxes->size(); i++)
    {
        cv::Mat roi = srcImg((*selected_boxes)[i]);
        cv::Mat replaceroi = replaceImg((*selected_boxes)[i]);
        cv::Mat binRoi = binariseBox(srcImg, (*selected_boxes)[i]);

        binRoi = translationPreProcess(&binRoi);

        getTranslation(&binRoi, &max_class, &argMax_class);

        binRoi = translationPreProcess(&binRoi);

        overlayTranslation(roi, replaceroi, &max_class, &argMax_class, option);

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "total time %f", duration.count()/1000.0);
    return replaceImg;


        //cv::rectangle(src, (*selected_boxes)[i], cv::Scalar(255, 0, 0), 1);
//        if (option == 1)
//        {
//            displayOverlay(srcImg, (*selected_boxes)[i], overlay, option);
//        }
//        else
//        {
//            displayOverlay(srcImg, (*selected_boxes)[i], srcImg, option);
//        }



//    cv::Mat img;
//    cvtColor(srcImg, img, cv::COLOR_RGBA2BGR);
//
//    cv::Mat grayImg;
//    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    /*
     * Test that image is correctly converted to grayscale
     */

//    cv::Mat grayDilate;
//    cv::Mat grayErode;
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
//    cv::dilate(grayImg, grayDilate, kernel, cv::Point(-1, -1), 1);
//    cv::erode(grayDilate, grayErode, kernel, cv::Point(-1, -1), 1);
//
//
//    cv::Mat graySmoothed;
//    cv::medianBlur(grayErode, graySmoothed, 5);
//
//    grayDilate.release();
//    grayErode.release();



    /*
     * Test that image is different after preprocessing applied
     */

//    cv::Mat grayBGR;
//    cvtColor(graySmoothed, grayBGR, cv::COLOR_GRAY2BGR);
//    graySmoothed.release();

    /*
    * Test that image is correctly converted to BGR and all three vec values are the same
    */



//    cv::Mat detectionImg;
//    detectionImg = Detection(grayBGR, img, option);
}

cv::Mat captureBoxImage(AAssetManager* manager, cv::Mat srcImg, int x, int y, int w, int h) {
    auto beg = std::chrono::high_resolution_clock::now();
    mgr = manager;
    cv::Rect box = cv::Rect(x, y, w, h);

    cv::Mat img;
    cvtColor(srcImg, img, cv::COLOR_RGBA2BGR);

    cv::Mat translateBox;
    displayOverlay(img, box, img, 0);

    cv::Mat translateFinal;
    cvtColor(img, translateFinal, cv::COLOR_BGR2RGBA);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "total time %f", duration.count()/1000.0);
    return translateFinal;
}


/*MSER Legacy code
 * LEFT IN FOR VIEWING PURPOSES BUT THIS CODE IS UNUSED WITHIN THE ACTUAL APPLICATION
 * THIS IS A PREVIOUS VERSION OF A DETECTION MODEL TO SHOW IMPLEMENTATION OF TRADITIONAL IMAGE DETECTION TECHNIQUES
 */
//void clearSuspiciousBoxes(cv::Mat& img, std::vector<cv::Rect> inBoxes, std::vector<cv::Rect>& outboxes, double suspicionThresh = 0.5, int widthSuspicion = 5, int heightSuspicion = 5, double aspectRatioSuspicion = 8.0)
//{
//    int height = img.size().height;
//    int width = img.size().width;
//
//    bool suspiciouslyLarge = 0;
//    bool suspiciouslyNarrow = 0;
//    bool suspiciousAspect = 0;
//
//    double aspectRatio;
//
//    for (auto box : inBoxes)
//    {
//        suspiciouslyLarge = box.width > width * suspicionThresh or box.height > height * suspicionThresh;
//        suspiciouslyNarrow = box.width <= widthSuspicion or box.height <= heightSuspicion;
//        aspectRatio = double(box.width) / double(box.height);
//        suspiciousAspect = aspectRatio >= aspectRatioSuspicion or aspectRatio <= 1 / aspectRatioSuspicion;
//
//        if (!(suspiciouslyLarge) && !(suspiciouslyNarrow) && !(suspiciousAspect))
//        {
//            outboxes.push_back(box);
//        }
//    }
//}
//
//void mergeBounding(std::vector<cv::Rect>& inBoxes, cv::Mat& img, std::vector<cv::Rect>& outBoxes, cv::Size scaleFactor)
//{
//    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1); // Create a blank image that we can draw rectangles on.
//    cv::Scalar colour = cv::Scalar(255);
//
//    //Draw filled version of our bounding boxes on mask image. This will give us connected bounding boxes we can find contours on to combine.
//    for (int i = 0; i < inBoxes.size(); i++)
//    {
//        cv::Rect bbox = inBoxes.at(i) + scaleFactor;
//        rectangle(mask, bbox, colour, cv::FILLED);
//
//    }
//
//    std::vector<std::vector<cv::Point>> contours;
//    //Draw contours on image and join them to then find our new bounding boxes.
//    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    for (int i = 0; i < contours.size(); i++)
//    {
//        outBoxes.push_back(cv::boundingRect(contours.at(i)) - scaleFactor);
//    }
//}
//
//
//cv::Mat mserDetection(cv::Mat img, cv::Mat colImg, bool thresholding = false, int xthresh = 10, int ythresh = 10)
//{
//    std::vector<std::vector<cv::Point>> regions;
//    std::vector<cv::Rect> boxes;
//
//    cv::Ptr<cv::MSER> mser = cv::MSER::create(7, 60, 14400, 0.25);
//
//    mser->detectRegions(img, regions, boxes);
//    cv::Scalar colour = cv::Scalar(255);
//
//    std::vector<cv::Rect> bboxes;
//
//    //This section removes any suspicious bounding boxes that are either too big or too small!
//
//    clearSuspiciousBoxes(img, boxes, bboxes);
//
//    //Below is the code to combine overlapping or close bounding boxes together
//
//    cv::Size scaleFactor(-10, -10); //Can adjust sensitivity of the boxes to other boxes by editing these values.
//    std::vector<cv::Rect> outboxes; //List of end rectangles that are retrieved
//
//    mergeBounding(bboxes, img, outboxes, scaleFactor);
//
//    double diff;
//    for (int i = 0; i < outboxes.size(); i++)
//    {
//        double aspectRatio = double(outboxes.at(i).width) / double(outboxes.at(i).height);
//
//        if (aspectRatio >= 2.0)
//        {
//            diff = double(outboxes.at(i).width) - double(outboxes.at(i).height);
//            outboxes[i] = outboxes.at(i) + cv::Size(0, diff / 4.0);
//        }
//        else if (aspectRatio <= (1.0 / 2.0))
//        {
//            diff = double(outboxes.at(i).height) - double(outboxes.at(i).width);
//            outboxes[i] = outboxes.at(i) + cv::Size(diff / 4.0, 0);
//        }
//    }
//
//    std::vector<cv::Rect> finalBoxes;
//
//    mergeBounding(outboxes, img, finalBoxes, cv::Size(0, 0));
//
//    cvtColor(img, img, cv::COLOR_GRAY2BGR);
//
//    auto beg = std::chrono::high_resolution_clock::now();
//    for (size_t i = 0; i < finalBoxes.size(); i++)
//    {
//        //rectangle(colImg, finalBoxes[i].tl(), finalBoxes[i].br(), cv::Scalar(0, 0, 255), 2);
//
//        // add correct overlay to colImg for this bounding box
//        displayOverlay(colImg, finalBoxes[i], colImg, 0);
//    }
//
//    return colImg;
//}
//
//cv::Mat gammaCorrect(cv::Mat img, double gam)
//{
//    cv::Mat hsvImg;
//    cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
//
//    std::vector<cv::Mat> vec_channels;
//    cv::split(hsvImg, vec_channels);
//
//    double mid = 0.5;
//    double mean = cv::mean(vec_channels[2])[0];
//    double gamma = log(mid * 255) / log(mean);
//
//    cv::Mat1d channel_gamma;
//
//    vec_channels[2].convertTo(channel_gamma, CV_64F);
//
//    cv::pow(channel_gamma, gam, channel_gamma);
//
//    channel_gamma.convertTo(vec_channels[2], CV_8U);
//
//    cv::merge(vec_channels, hsvImg);
//
//    cvtColor(hsvImg, img, cv::COLOR_HSV2BGR);
//
//    return img;
//
//}
