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
#include "ncnn.h"

ncnn::Net translationModel;
ncnn::Net detectionModel;
bool modelInitialisedFlag = false;
bool detmodelInitialisedFlag = false;
AAssetManager* mgr;

cv::Mat binariseBox(cv::Mat img, cv::Rect inBox)
{
    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);


    cv::Mat boxImg(grayImg, inBox);
    cv::Mat threshBox;

    cv::threshold(boxImg, threshBox, 0, 255, cv::THRESH_OTSU);
    boxImg.release();
    grayImg.release();

    int rows = threshBox.rows;
    int cols = threshBox.cols;

    int box_tl = (int) threshBox.at<uchar>(0, 0);
    int box_bl = (int) threshBox.at<uchar>(rows - 1, 0);
    int box_tr = (int) threshBox.at<uchar>(0, cols-1);
    int box_br = (int) threshBox.at<uchar>(rows - 1, cols - 1);

    int sum_corners = box_tl + box_bl + box_tr + box_br;

    if (sum_corners <= 255)
    {
        cv::bitwise_not(threshBox, threshBox);
    }

    cv::Mat threshBGR;
    cvtColor(threshBox, threshBGR, cv::COLOR_GRAY2BGR);

    return threshBGR;
}

void loadTranslationModel() {
    // Load model
    int ret = translationModel.load_param(mgr,"seals-resnet50-sim-opt.param");
    if (ret) {
         __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "seals-resnet50-sim-opt.bin");
    if (ret) {
       __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    modelInitialisedFlag = true;
}

void preloadModels(AAssetManager* manager) {

    mgr = manager;

    int ret = translationModel.load_param(mgr,"seals-resnet50-sim-opt.param");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "seals-resnet50-sim-opt.bin");
    if (ret) {
        __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    modelInitialisedFlag = true;

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


void displayOverlay(cv::Mat colImg, cv::Rect location){

    std::string bugString;

    if(!modelInitialisedFlag)
    {
        loadTranslationModel();
    }

    // get input from bounding box
    cv::Rect roiRect(location);
    cv::Mat roi = colImg(roiRect);
    // binarise image
    cv::Mat binRoi = binariseBox(colImg, roiRect);

    bugString = "Retrieved binary box";
    __android_log_print(ANDROID_LOG_DEBUG, "binary box", "%s", bugString.c_str());

    cv::Mat binRoiR;
    cv::resize(binRoi, binRoiR, cv::Size(232, 232));

    const int cropSize = 224;
    const int offsetW = (binRoiR.cols - cropSize) / 2;
    const int offsetH = (binRoiR.rows - cropSize) / 2;
    const cv::Rect roiBin(offsetW, offsetH, cropSize, cropSize);
    binRoi = binRoiR(roiBin).clone();

    binRoiR.release();

    // Convert image data to ncnn format
    // opencv image in bgr, model needs rgb
    ncnn::Mat input = ncnn::Mat::from_pixels(binRoi.data, ncnn::Mat::PIXEL_BGR2RGB, binRoi.cols, binRoi.rows);

    float means[] = {0.485, 0.456, 0.406};
    float norms[] = {0.229, 0.224, 0.225};

    input.substract_mean_normalize(means, norms);

    // Inference
    ncnn::Extractor extractor = translationModel.create_extractor();
    extractor.set_light_mode(true);
    extractor.input("input.1", input);
    ncnn::Mat output;
    extractor.extract("503", output);

    float max = output[0];
    std::string argMax;
    for (int j=0; j<output.w; j++) {
        if (output[j] > max){
            max = output[j];
            argMax = std::to_string(j);;
        }
    }

    // check for threshold TODO

   // get file name
    std::string filename = "overlays/";
    filename.append(argMax);
    filename.append(".bmp");
    // load file from assets
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), 0);
    long size = AAsset_getLength(asset);
    uchar* buffer = (uchar*) malloc (sizeof(uchar)*size);
    AAsset_read (asset,buffer,size);
    AAsset_close(asset);

    // convert file to rgb image
    cv::Mat rawData( 1, size, CV_8UC1, (void*)buffer);
    cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);

    cv::Mat decodeColor;
    cvtColor(decodedImage,decodeColor, cv::COLOR_BGR2RGB);
    decodedImage.release();

    //overlay image on rectangle

    cv::Mat overlayImg;
    cv::resize(decodeColor, overlayImg, roi.size());
    decodeColor.release();

    addWeighted(overlayImg, 1, roi, 0, 0, roi);
    overlayImg.release();
    //addWeighted(decodedImage, 1, binRoi, 0, 0, binRoi);
}



void clearSuspiciousBoxes(cv::Mat& img, std::vector<cv::Rect> inBoxes, std::vector<cv::Rect>& outboxes, double suspicionThresh = 0.5, int widthSuspicion = 5, int heightSuspicion = 5, double aspectRatioSuspicion = 8.0)
{
    int height = img.size().height;
    int width = img.size().width;

    bool suspiciouslyLarge = 0;
    bool suspiciouslyNarrow = 0;
    bool suspiciousAspect = 0;

    double aspectRatio;

    for (auto box : inBoxes)
    {
        suspiciouslyLarge = box.width > width * suspicionThresh or box.height > height * suspicionThresh;
        suspiciouslyNarrow = box.width <= widthSuspicion or box.height <= heightSuspicion;
        aspectRatio = double(box.width) / double(box.height);
        suspiciousAspect = aspectRatio >= aspectRatioSuspicion or aspectRatio <= 1 / aspectRatioSuspicion;

        if (!(suspiciouslyLarge) && !(suspiciouslyNarrow) && !(suspiciousAspect))
        {
            outboxes.push_back(box);
        }
    }
}

void mergeBounding(std::vector<cv::Rect>& inBoxes, cv::Mat& img, std::vector<cv::Rect>& outBoxes, cv::Size scaleFactor)
{
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1); // Create a blank image that we can draw rectangles on.
    cv::Scalar colour = cv::Scalar(255);

    //Draw filled version of our bounding boxes on mask image. This will give us connected bounding boxes we can find contours on to combine.
    for (int i = 0; i < inBoxes.size(); i++)
    {
        cv::Rect bbox = inBoxes.at(i) + scaleFactor;
        rectangle(mask, bbox, colour, cv::FILLED);

    }

    std::vector<std::vector<cv::Point>> contours;
    //Draw contours on image and join them to then find our new bounding boxes.
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++)
    {
        outBoxes.push_back(cv::boundingRect(contours.at(i)) - scaleFactor);
    }
}


cv::Mat mserDetection(cv::Mat img, cv::Mat colImg, bool thresholding = false, int xthresh = 10, int ythresh = 10)
{
    std::vector<std::vector<cv::Point>> regions;
    std::vector<cv::Rect> boxes;

    cv::Ptr<cv::MSER> mser = cv::MSER::create(7, 60, 14400, 0.25);

    mser->detectRegions(img, regions, boxes);
    cv::Scalar colour = cv::Scalar(255);

    std::vector<cv::Rect> bboxes;

    //This section removes any suspicious bounding boxes that are either too big or too small!

    clearSuspiciousBoxes(img, boxes, bboxes);

    //Below is the code to combine overlapping or close bounding boxes together

    cv::Size scaleFactor(-10, -10); //Can adjust sensitivity of the boxes to other boxes by editing these values.
    std::vector<cv::Rect> outboxes; //List of end rectangles that are retrieved

    mergeBounding(bboxes, img, outboxes, scaleFactor);

    double diff;
    for (int i = 0; i < outboxes.size(); i++)
    {
        double aspectRatio = double(outboxes.at(i).width) / double(outboxes.at(i).height);

        if (aspectRatio >= 2.0)
        {
            diff = double(outboxes.at(i).width) - double(outboxes.at(i).height);
            outboxes[i] = outboxes.at(i) + cv::Size(0, diff / 4.0);
        }
        else if (aspectRatio <= (1.0 / 2.0))
        {
            diff = double(outboxes.at(i).height) - double(outboxes.at(i).width);
            outboxes[i] = outboxes.at(i) + cv::Size(diff / 4.0, 0);
        }
    }

    std::vector<cv::Rect> finalBoxes;

    mergeBounding(outboxes, img, finalBoxes, cv::Size(0, 0));

    cvtColor(img, img, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < finalBoxes.size(); i++)
    {
        //rectangle(colImg, finalBoxes[i].tl(), finalBoxes[i].br(), cv::Scalar(0, 0, 255), 2);

        // add correct overlay to colImg for this bounding box
        displayOverlay(colImg, finalBoxes[i]);
    }

    return colImg;
}

cv::Mat gammaCorrect(cv::Mat img, double gam) {


    cv::Mat hsvImg;
    cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> vec_channels;
    cv::split(hsvImg, vec_channels);

    double mid = 0.5;
    double mean = cv::mean(vec_channels[2])[0];
    double gamma = log(mid * 255) / log(mean);

    cv::Mat1d channel_gamma;

    vec_channels[2].convertTo(channel_gamma, CV_64F);

    cv::pow(channel_gamma, gam, channel_gamma);

    channel_gamma.convertTo(vec_channels[2], CV_8U);

    cv::merge(vec_channels, hsvImg);

    cvtColor(hsvImg, img, cv::COLOR_HSV2BGR);

    return img;

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
    std::vector<float>& score_ref = *scores;

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

cv::Mat Detection(cv::Mat src, cv::Mat orig) {

    std::string bugString;

    cv::Mat srcScaled;
    float sf;
    bool r_or_c = false;

    if(src.rows >= src.cols)
    {
        sf = 512.0/(float)src.rows;
        r_or_c = false;
    }
    else
    {
        sf = 512.0/(float)src.cols;
        r_or_c = true;
    }

    cv::resize(src, srcScaled, cv::Size(), sf, sf);

    cv::Mat srcFinal;

    if (r_or_c)
    {
        cv::copyMakeBorder(srcScaled, srcFinal, 0, 512-srcScaled.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }
    else
    {
        cv::copyMakeBorder(srcScaled, srcFinal, 0, 0, 0, 512-srcScaled.cols, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    }

//    // Load model
//    ncnn::Net net;
//    int ret = net.load_param("model.ncnn.param");
//    if (ret) {
//        // __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
//        std::cout << "Failed to load the model parameters" << std::endl;
//    }
//    ret = net.load_model("model.ncnn.bin");
//    if (ret) {
//        //__android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
//        std::cout << "Failed to load the model weights" << std::endl;
//    }

    // Convert image data to ncnn format
    // opencv image in bgr, model needs rgb
    if(!detmodelInitialisedFlag){
        loadDetectionModel();
    }

    bugString = "Height: " + std::to_string(srcFinal.rows);
    __android_log_print(ANDROID_LOG_DEBUG, "resize", "%s", bugString.c_str());
    bugString = "Width: " + std::to_string(srcFinal.cols);
    __android_log_print(ANDROID_LOG_DEBUG, "resize", "%s", bugString.c_str());

    ncnn::Mat input = ncnn::Mat::from_pixels(srcFinal.data,
                                             ncnn::Mat::PixelType::PIXEL_RGB,
                                             srcFinal.cols, srcFinal.rows);


    float means[] = {0.0, 0.0, 0.0};
    float norms[] = {1.0/255.0, 1.0/255.0, 1.0/255.0};

    input.substract_mean_normalize(means, norms);

    // Inference
    ncnn::Extractor extractor = detectionModel.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat output;
    extractor.extract("out0", output);

    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);
    std::vector<cv::Rect>* boxes = new std::vector<cv::Rect>();
    std::vector<float>* confidences = new std::vector<float>();

    int sec_size = out_flatterned.w/5;
    for (int j=0; j<sec_size; j++)
    {
        if (out_flatterned[j+(sec_size*4)] > 0.5)
        {
            confidences->push_back(float(out_flatterned[j+(sec_size*4)]));

            float x = out_flatterned[j] / sf;
            float y = out_flatterned[j+(sec_size)] / sf;
            float w = out_flatterned[j+(sec_size*2)] / sf;
            float h = out_flatterned[j+(sec_size*3)] / sf;

            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            int width = int(w);
            int height = int(h);
            boxes->push_back(cv::Rect(left, top, width, height));
        }
        else
        {
            bugString = "Confidence: " + std::to_string(out_flatterned[j+(sec_size*4)]);
            //__android_log_print(ANDROID_LOG_DEBUG, "det_boxes", "%s", bugString.c_str());
        }
    }

    bugString = "Detected " + std::to_string(boxes->size()) + " boxes";
    __android_log_print(ANDROID_LOG_DEBUG, "det_boxes", "%s", bugString.c_str());

    std::vector<cv::Rect>* selected_boxes = new std::vector<cv::Rect>();
    nms(boxes, confidences, selected_boxes, 0.5);

    bugString = "NMS " + std::to_string(selected_boxes->size()) + " boxes";
    __android_log_print(ANDROID_LOG_DEBUG, "det_boxes", "%s", bugString.c_str());

    for (std::size_t i = 0; i != selected_boxes->size(); i++)
    {
        //cv::rectangle(src, (*selected_boxes)[i], cv::Scalar(255, 0, 0), 1);
        displayOverlay(orig, (*selected_boxes)[i]);
    }

    return orig;

}

cv::Mat captureImage(AAssetManager* manager, cv::Mat srcImg) {

    mgr = manager;

    cv::Mat img;
    cvtColor(srcImg, img, cv::COLOR_RGBA2BGR);

    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    //img = gammaCorrect(img, 0.95);

    cv::Mat grayDilate;
    cv::Mat grayErode;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(grayImg, grayDilate, kernel, cv::Point(-1, -1), 1);
    cv::erode(grayDilate, grayErode, kernel, cv::Point(-1, -1), 1);


    cv::Mat graySmoothed;
    cv::medianBlur(grayErode, graySmoothed, 5);

    grayDilate.release();
    grayErode.release();
    grayImg.release();

//    cv::Mat mserDetect;
//    mserDetect = mserDetection(graySmoothed, img, false);
//
//    return mserDetect;

    cv::Mat grayBGR;
    cvtColor(graySmoothed, grayBGR, cv::COLOR_GRAY2BGR);

    graySmoothed.release();

    cv::Mat detectionImg;
    detectionImg = Detection(img, img);

    cv::Mat detectionFinal;
    cvtColor(detectionImg, detectionFinal, cv::COLOR_BGR2RGBA);

    return detectionFinal;
}

//int main(int, char**) {
//    std::string path = "..\\..\\..\\seal script image 14.jpg";
//    cv::Mat img = cv::imread(path);
//    double factor = 700.0 / img.size().height;
//    cv::resize(img, img, cv::Size(), factor, factor, cv::INTER_CUBIC);
//
//
//
//
//    cv::Mat Image = captureImage(img);
//
//    cv::imshow("Image", Image);
//
//    cv::waitKey(0);
//}
