#include "enhance.h"
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "iostream"

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
        else
        {
            //std::cout << aspectRatio << "\n";
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
        outBoxes.push_back(cv::boundingRect(contours.at(i))-scaleFactor);
    }
}

/*cv::RNG rng(12345);
for (size_t i = 0; i < contours.size(); i++)
{
    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(mask2, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
}

cv::imshow("Contours", mask2);*/

/*for (int i = 0; i < contours.size(); i++)
{
    double area = cv::contourArea(contours.at(i));
    double perimeter = cv::arcLength(contours.at(i), 1);
    double squareness = cv::norm(((perimeter / 4) * (perimeter / 4)) - area);
}*/


cv::Mat mserDetection(cv::Mat img, bool thresholding = false, int xthresh = 10, int ythresh = 10)
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

        //std::cout << aspectRatio << "\n";
        if (aspectRatio >= 2.0)
        {
            //std::cout << "Detected!\n";
            diff = double(outboxes.at(i).width) - double(outboxes.at(i).height);
            outboxes[i] = outboxes.at(i) + cv::Size(0, diff/4.0);
        }
        else if (aspectRatio <= (1.0/2.0))
        {
            //std::cout << "Detected!\n";
            diff = double(outboxes.at(i).height) - double(outboxes.at(i).width);
            outboxes[i] = outboxes.at(i) + cv::Size(diff/4.0, 0);
        }
    }

    std::vector<cv::Rect> finalBoxes;

    mergeBounding(outboxes, img, finalBoxes, cv::Size(-2, -2));


    cvtColor(img, img, cv::COLOR_GRAY2BGR);


    for (size_t i = 0; i < finalBoxes.size(); i++)
    {
        rectangle(img, finalBoxes[i].tl(), finalBoxes[i].br(), cv::Scalar(0,0,255), 2);
    }

    //This section can be uncommented to draw original bounding boxes from MSER instead of the updated ones.
    //for (size_t i = 0; i < bboxes.size(); i++)
    //{
    //    rectangle(img, bboxes[i].tl(), bboxes[i].br(), colour, 2);
    //}

    return img;
}

void gammaCorrect(cv::Mat img, double gam) {
    cv::Mat1d dimg;

    img.convertTo(dimg, CV_64F);

    cv::Mat1d dgam;
    cv::pow(dimg, gam, dgam);

    dgam.convertTo(img, CV_8U);
}

// altered function prototype
cv::Mat showImage2(cv::Mat img){
    //read in image
    //read in image - this is my path to it you'll need to change that for yourself.
    //std::string path = "..\\..\\..\\seal script image 14.jpg";
    //cv::Mat img = cv::imread(path);

    //Use this to resize final image
    //cv::resize(img, img, cv::Size(), 1.2, 1.2, cv::INTER_CUBIC);

    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    gammaCorrect(grayImg, 0.95);

    //kernel = np.ones((1, 1), np.uint8)
    cv::Mat grayDilate;
    cv::Mat grayErode;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
    cv::dilate(grayImg, grayDilate, kernel, cv::Point(-1,-1), 1);
    cv::erode(grayDilate, grayErode, kernel, cv::Point(-1, -1), 1);

    //apply bilateral/median filter to reduce noise
    //Median filter seems to work better on the whole. Edges of the characters don't seem to be too affected in the blurring process.
    cv::Mat graySmoothed;
    cv::medianBlur(grayErode, graySmoothed, 3);
    //cv::bilateralFilter(grayErode, graySmoothed, 5, 90, 90);

    //cv::Mat grayThresholded;
    //cv::adaptiveThreshold(graySmoothed, grayThresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 2);

    cv::Mat mserDetect;
    mserDetect = mserDetection(graySmoothed, false);
    //cvtColor(mserDetect, mserDetect, cv::COLOR_GRAY2BGR);

    //cv::imshow("Enhanced Image", graySmoothed);
    //cv::imshow("Bounding boxes with MSER", mserDetect);

    //cv::waitKey(0);
    return mserDetect;
}

//sharpen LAB image -> Histogram equalize -> Grayscale -> Alpha-beta correction -> Bilateral Filter -> Gamma correction -> Canny edge detection
//Thinking we should swap it to: Fourier Transform -> Sharpen and remove noise frequencies and then see what other steps will be useful from there

/*
hyperparameters which need to be decided on:
-gamma value
-alpha beta ratio (contrast increase/decrease)
-canny thresholds
-bilteral filtering sigma values and diameter
-unsharp mask, gaussian sigma and amount
-thresholding for binarization
*/

void abCorrection(cv::Mat img, double minBright) {
    //get the average brightness value across the image
    double avgBright = cv::mean(img)[0] / 255;

    std::cout << avgBright << "\n";

    double ratio = avgBright / minBright;

    std::cout << ratio << "\n";

    if (ratio >= 1)
    {
        //std::cout << "Image already bright enough\n";
        return;
    }

    cv::convertScaleAbs(img, img, (1 / ratio), 0);
}


int showImage() {
    //read in image
    //width and height are constants for window size of images
    int WIDTH = 650;
    int HEIGHT = 650;
    //read in image - this is my path to it you'll need to change that for yourself.
    std::string path = "..\\..\\..\\seal script image 13.jpg";
    cv::Mat img = cv::imread(path);
    //Use this to resize final image
    cv::resize(img, img, cv::Size(), 1, 1, cv::INTER_CUBIC);

    //convert original image to Lab - this is a good format for sharpening
    cv::Mat labImg;
    cvtColor(img, labImg, cv::COLOR_BGR2Lab);

    //split Lab into channels
    std::vector<cv::Mat> vec_channels_lab;
    cv::split(labImg, vec_channels_lab);

    //perform sharpening on Lightness channel.
    double sigma = 8, amount = 1;
    cv::Mat blurry;
    GaussianBlur(vec_channels_lab[0], blurry, cv::Size(3, 3), sigma);
    addWeighted(vec_channels_lab[0], 1 + amount, blurry, -amount, 0, vec_channels_lab[0]);

    cv::merge(vec_channels_lab, labImg);

    cvtColor(labImg, img, cv::COLOR_Lab2BGR);

    //convert image to YCrCb so histogram equalising doesn't affecting image colouring
    cv::Mat hist_equalized_img;
    cvtColor(img, hist_equalized_img, cv::COLOR_BGR2YCrCb);

    //split YCrCb into channels and histogram equalise Y channel.
    std::vector<cv::Mat> vec_channels;
    cv::split(hist_equalized_img, vec_channels);
    cv::equalizeHist(vec_channels[0], vec_channels[0]);
    cv::merge(vec_channels, hist_equalized_img);

    //convert image back to RGB
    cvtColor(hist_equalized_img, hist_equalized_img, cv::COLOR_YCrCb2BGR);

    //create a grayscale version of histogram equalised image
    cv::Mat grayImg;
    cv::cvtColor(hist_equalized_img, grayImg, cv::COLOR_BGR2GRAY);

    //perform alpha-beta correction to increase contrast
    abCorrection(grayImg, 0.8);

    //apply bilateral filter to reduce noise
    cv::Mat graySmoothed;
    cv::bilateralFilter(grayImg, graySmoothed, 5, 115, 30);

    //perform gamma correction to decrease brightness
    gammaCorrect(graySmoothed, 0.885);

    //cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    //cv::resize(img, img, cv::Size(), 0.75, 0.75);
    //cv::Mat imSmoothed;
    //cv::bilateralFilter(img, imSmoothed, 9, 60, 60);

    //currently not in use, thinking about how thresholding can actually be used if text is white and black - needs to be universal!
    cv::Mat invertGray;
    cv::bitwise_not(graySmoothed, invertGray);

    cv::Mat thresh;
    cv::threshold(invertGray, thresh, 110, 255, 3);
    cv::bitwise_not(thresh, thresh);

    //perhaps implement frequency filtering for noise?


    //perform canny edge detection to attempt to detect character edges.
    cv::Mat canny;
    cv::Canny(graySmoothed, canny, 200 / 3, 200);

    cv::Mat mserDetect;
    mserDetect = mserDetection(graySmoothed, false);

    //perform canny edge detection on original sharpened image. Not even grayscale.
    //cv::Mat cannyImg;
    //cv::Canny(img, cannyImg, 200 / 3, 200);

    //perform canny edge on Lab image that was sharpened - just for testing.
    //cv::Mat cannySharp;
    //cv::Canny(labImg, cannySharp, 200 / 3, 200);


    //This is all contouring stuff that is irrelevant rn. Don't worry about this.

    //std::vector<std::vector<cv::Point> > contours;
    //std::vector<cv::Vec4i> hierarchy;
    //findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    //cv::Mat drawing = cv::Mat::zeros(canny.size(), CV_8UC3);

    //cv::RNG rng(12345);

    //for (size_t i = 0; i < contours.size(); i++)
    //{
    //    cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    //    drawContours(drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
    //}

    // show the image on the window
    //cv::imshow("Original Gray Image", img);
    //cv::imshow("Original Gray Image - Smoothed", imSmoothed);
    //cv::imshow("Hist Equalised Image", hist_equalized_img);

    //for some reason - resizing to fit a window generates more noise so these have been commented out for now as well.

    //cv::namedWindow("Sharp", cv::WINDOW_NORMAL);
    //cv::namedWindow("Enhanced Grayscale Image", cv::WINDOW_NORMAL);
    //cv::namedWindow("Canny Edge Detector on Enhanced Image", cv::WINDOW_NORMAL);
    //cv::namedWindow("Canny on Sharp Image", cv::WINDOW_NORMAL);
    //cv::namedWindow("Threshold", cv::WINDOW_NORMAL);

    //cv::resizeWindow("Sharp", WIDTH, HEIGHT);
    //cv::resizeWindow("Enhanced Grayscale Image", WIDTH, HEIGHT);
    //cv::resizeWindow("Canny Edge Detector on Enhanced Image", WIDTH, HEIGHT);
    //cv::resizeWindow("Canny on Sharp Image", WIDTH, HEIGHT);
    //cv::resizeWindow("Threshold", WIDTH, HEIGHT);



    //cv::imshow("Sharp", labImg);
    //cv::imshow("Enhanced Grayscale Image", graySmoothed);
    //cv::imshow("Canny Edge Detector on Enhanced Image", canny);
    //cv::imshow("Bounding boxes with MSER", mserDetect);
    //cv::imshow("Canny Edge on Original Image", cannyImg);
    //cv::imshow("Canny on Sharp Image", cannySharp);
    //cv::imshow("Threshold", thresh);
    //cv::imshow("Contours", drawing);

//    cv::waitKey(0);
    return 0;
}

/*int showWebCameraContent() {
    // open the first webcam plugged in the computer
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "ERROR: Could not open camera" << std::endl;
        return 1;
    }

    // create a window to display the images from the webcam
    cv::namedWindow("Webcam");

    // this will contain the image from the webcam
    cv::Mat frame;

    // display the frame until you press a key
    while (1) {
        // capture the next frame from the webcam
        camera >> frame;

        //flip orientation so it looks right
        cv::flip(frame, frame, 1);

        cv::Mat hist_equalized_frame;
        cvtColor(frame, hist_equalized_frame, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> vec_channels;
        cv::split(hist_equalized_frame, vec_channels);

        cv::equalizeHist(vec_channels[0], vec_channels[0]);


        cv::merge(vec_channels, hist_equalized_frame);

        cvtColor(hist_equalized_frame, hist_equalized_frame, cv::COLOR_YCrCb2BGR);

        //cv::Mat contours;
        //cv::Canny(hist_equalized_frame, contours, 40, 120);

        cv::Mat thresh;

        cv::cvtColor(hist_equalized_frame, thresh, cv::COLOR_BGR2GRAY);

        cv::Mat thresh2;

        cv::threshold(thresh, thresh2, 100, 255, cv::THRESH_BINARY);



        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange[] = { range };

        cv::Mat grayHist;

        cv::calcHist(&thresh, 1, 0, cv::Mat(), grayHist, 1, &histSize, histRange, true, false);
        int hist_w = 512, hist_h = 400;

        int bin_w = cvRound((double)hist_w / histSize);

        cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

        cv::normalize(grayHist, grayHist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; i++)
        {
            line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(grayHist.at<float>(i - 1))),
                 cv::Point(bin_w * (i), hist_h - cvRound(grayHist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        }

        // show the image on the window
        cv::imshow("Normal Webcam", frame);
        cv::imshow("Hist Equalised Webcam", hist_equalized_frame);
        cv::imshow("Gray Webcam", thresh);
        cv::imshow("Threshold", thresh2);
        cv::imshow("Histogram", histImage);
        //cv::imshow("Contours", contours);

        // wait (10ms) for a key to be pressed
        if (cv::waitKey(10) >= 0) {
            break;
        }
    }
    return 0;
}*/

//int main(int, char**) {
//    showImage2();
//    //showWebCameraContent();
//}
