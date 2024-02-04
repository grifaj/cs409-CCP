#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"
#include <math.h>

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

    std::string path = "..\\..\\..\\smiley.jpg";
    cv::Mat smiley = cv::imread(path);
    
    for (size_t i = 0; i < finalBoxes.size(); i++)
    {
        rectangle(colImg, finalBoxes[i].tl(), finalBoxes[i].br(), cv::Scalar(0, 0, 255), 2);

        /*cv::Point bottomLeft = finalBoxes[i].tl() + cv::Point(0, finalBoxes[i].height);*/
        cv::resize(smiley, smiley, cv::Size(finalBoxes[i].width, finalBoxes[i].height), 0, 0, cv::INTER_CUBIC);


        cv::Mat insetImage(colImg, finalBoxes[i]);
        smiley.copyTo(insetImage);

        //cv::putText(colImg,
        //    "A",
        //    bottomLeft, // Coordinates (Bottom-left corner of the text string in the image)
        //    cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
        //    5.0, // Scale. 2.0 = 2x bigger
        //    cv::Scalar(0, 0, 0), // BGR Color
        //    1);

        

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

cv::Mat captureImage(cv::Mat img) {

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

int main(int, char**) {
    std::string path = "..\\..\\..\\seal script image 14.jpg";
    cv::Mat img = cv::imread(path);
    double factor = 700.0 / img.size().height;
    cv::resize(img, img, cv::Size(), factor, factor, cv::INTER_CUBIC);

    


    cv::Mat Image = captureImage(img);

    cv::imshow("Image", Image);

    cv::waitKey(0);
}