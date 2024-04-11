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
bool modelInitilisedFlag = false;
AAssetManager* mgr;

cv::Mat binariseBox(cv::Mat img, cv::Rect inBox)
{
    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);


    cv::Mat boxImg(grayImg, inBox);
    cv::Mat threshBox;

    cv::threshold(boxImg, threshBox, 0, 255, cv::THRESH_OTSU);

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

    return threshBox;
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
    modelInitilisedFlag = true;
}


void displayOverlay(cv::Mat colImg, cv::Rect location){

    if(!modelInitilisedFlag){
        loadTranslationModel();
    }

    // get input from bounding box
    cv::Rect roiRect(location);
    cv::Mat roi = colImg(roiRect);
    // binarise image
    cv::Mat binRoi = binariseBox(colImg, roiRect);

    // Convert image data to ncnn format
    // opencv image in bgr, model needs rgb
    ncnn::Mat input = ncnn::Mat::from_pixels(roi.data, ncnn::Mat::PIXEL_BGR2RGB, roi.cols, roi.rows);

    // Inference
    ncnn::Extractor extractor = translationModel.create_extractor();
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
    cv::Mat decodedImage  =  imdecode(rawData, cv::IMREAD_COLOR);
    cvtColor(decodedImage,decodedImage, cv::COLOR_BGR2RGB);

    //overlay image on rectangle
    resize(decodedImage, decodedImage, roi.size());
    //addWeighted(decodedImage, 1, roi, 0, 0, roi);
    addWeighted(decodedImage, 1, binRoi, 0, 0, binRoi);
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
        rectangle(colImg, finalBoxes[i].tl(), finalBoxes[i].br(), cv::Scalar(0, 0, 255), 2);

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

cv::Mat captureImage(AAssetManager* manager, cv::Mat img) {

    mgr = manager;

    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    img = gammaCorrect(img, 0.95);

    cv::Mat grayDilate;
    cv::Mat grayErode;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(grayImg, grayDilate, kernel, cv::Point(-1, -1), 1);
    cv::erode(grayDilate, grayErode, kernel, cv::Point(-1, -1), 1);

    cv::Mat graySmoothed;
    cv::medianBlur(grayErode, graySmoothed, 5);

    cv::Mat mserDetect;
    mserDetect = mserDetection(graySmoothed, img, false);

    return mserDetect;
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
