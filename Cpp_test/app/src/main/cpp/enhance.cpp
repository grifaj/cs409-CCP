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


//load translation model ncnn files into ncnn::net global variable
//only used if an error occurs and clears the preloaded models
void loadTranslationModel() {
    // Load model
    int ret = translationModel.load_param(mgr,"mobilenet_v3_large-sim-opt.param");
    if (ret)
    {
         __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "mobilenet_v3_large-sim-opt.bin");
    if (ret)
    {
       __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    modelInitialisedFlag = true;
}

//load detection model ncnn files into ncnn::net global variable
//only used if an error occurs and clears the preloaded models
void loadDetectionModel()
{
    int ret = detectionModel.load_param(mgr,"model.ncnn.param");
    if (ret)
    {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = detectionModel.load_model(mgr, "model.ncnn.bin");
    if (ret)
    {
        __android_log_print(ANDROID_LOG_ERROR, "load_weight_error", "Failed to load the model weights");
    }
    detmodelInitialisedFlag = true;
}

//This function is used to preload the translation and detection model on app startup, so that it doesn't have to
//be loaded when the user wants to detect
void preloadModels(AAssetManager* manager) {
    auto beg = std::chrono::high_resolution_clock::now();
    mgr = manager;

    int ret = translationModel.load_param(mgr,"mobilenet_v3_large-sim-opt.param");
    if (ret)
    {
        __android_log_print(ANDROID_LOG_ERROR, "load_param_error", "Failed to load the model parameters");
    }
    ret = translationModel.load_model(mgr, "mobilenet_v3_large-sim-opt.bin");
    if (ret)
    {
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

//takes in a pointer to an image, converts it from RGBA (this will be the android phone formatting)
//into a grayscale image. it then returns the grayscale version of the image.
cv::Mat grayImage(cv::Mat* srcImg)
{
    cv::Mat img;
    cvtColor(*srcImg, img, cv::COLOR_RGBA2BGR);

    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    return grayImg;
}

//this takes in a pointer to an image, applies morphological closing and median blur to it
//then returns the processed image
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

//takes in a gray image and converts it back into BGR formatting (it will still appear gray it
//just now has three channels). Returns BGR formatted gray image.
cv::Mat grayToBGR(cv::Mat* srcImg)
{
    cv::Mat grayBGR;
    cvtColor(*srcImg, grayBGR, cv::COLOR_GRAY2BGR);

    return grayBGR;
}

//this takes in an image and a scale factor variable to update.
//It attempts to resize the image so the longer dimension becomes 512px and the shorter
//dimension is scaled accordingly, the sf pointer is updated to keep this value for later use
//the scaled image is returned
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

//this takes an image and works out which side is shorter after the scaling that just happened
//then scales the shorter side up to 512px to produce a 512x512 image with aspect ratio maintained
//it returns the padded image
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

//this takes in a source image, the original colour image, the scale factor used to scale in resizeSF
//and a pointer to a vector containing bounding boxes and confidence scores of them
void detectModel(cv::Mat *srcImg, cv::Mat *orig, float* sf, std::vector<cv::Rect>* boxes, std::vector<float>* confidences)
{
    //convert the image we want to perform detection on into ncnn format so model can use it
    ncnn::Mat input = ncnn::Mat::from_pixels(srcImg->data,
                                             ncnn::Mat::PixelType::PIXEL_BGR,
                                             srcImg->cols, srcImg->rows);


    //normalise the image to mean of zero and std of 1
    float means[] = {0.0, 0.0, 0.0};
    float norms[] = {1.0/255.0, 1.0/255.0, 1.0/255.0};
    input.substract_mean_normalize(means, norms);

    //create extractor which takes input and gives output
    ncnn::Extractor extractor = detectionModel.create_extractor();
    extractor.input("in0", input);
    ncnn::Mat output;
    extractor.extract("out0", output);

    //flatten the output
    ncnn::Mat out_flattened = output.reshape(output.w * output.h * output.c);
    //outputs are given such that it gives all results of a specific type first and then goes
    //to the next type. i.e. all box x's, all box y's, all box widths, all box heights, all box confidences.



    //set up a confidence thresh to remove any boxes that aren't above this
    float confidenceThresh = 0.65;
    //because result is row major we have to set up a method to read across columns for individual
    //boxes and their scores. this is done by getting flattened shape divided by 5.
    //then for each iteration we look at this index, this index + sec_size and so on...
    int sec_size = out_flattened.w / 5;
    for (int j=0; j<sec_size; j++)
    {
        //if confidence score for this box is greater than confidence thresh
        if (out_flattened[j + (sec_size * 4)] > confidenceThresh)
        {
            //get x,y,w,h from our results for a specific box
            float x = out_flattened[j] / *sf;
            float y = out_flattened[j + (sec_size)] / *sf;
            float w = out_flattened[j + (sec_size * 2)] / *sf;
            float h = out_flattened[j + (sec_size * 3)] / *sf;

            //calculate top left of the box using what we have (x and y is center of box)
            int left = int((x - 0.5 * w));
            int top = int((y - 0.5 * h));

            //ensure that box co-ordinates do not exceed bounds
            if (left > 0 && top > 0 && w > 0 && h > 0 &&
                left+w < orig->cols && top+h < orig->rows)
            {
                //add this box to our boxes and its confidence to list of scores
                int width = int(w);
                int height = int(h);
                boxes->push_back(cv::Rect(left, top, width, height));
                confidences->push_back(float(out_flattened[j + (sec_size * 4)]));
            }
        }
    }
}

//this function takes in vector of boxes, and vector of their confidence scores
//sorts the boxes using confidence score vector as a key
void sortParallelVector(std::vector<cv::Rect>* vec, std::vector<float>* score_vec)
{
    //set up reference vectors to easily be able to index elements
    std::vector<cv::Rect>& vec_ref = *vec;
    std::vector<float>& score_ref = *score_vec;

    //set up some new vectors to help sort
    std::vector<std::size_t> index_vec;
    std::vector<cv::Rect> vec_ordered;
    std::vector<float> score_vec_ordered;

    //push back an index for each element we have in our vectors we want to sort
    for (std::size_t i = 0; i != vec->size(); ++i)
    {
        index_vec.push_back(i);
    }

    //sort the indexes we just put into the vector above so that the indexes are in order of which
    //index has the highest score in our score vector down to the lowest score in score vector
    std::sort(
            index_vec.begin(), index_vec.end(),
            [&](std::size_t a, std::size_t b) {return score_ref[a] > score_ref[b];});

    //go through our index list, that index from original lists into our new ordered lists
    for (std::size_t i = 0; i != index_vec.size(); ++i)
    {
        vec_ordered.push_back(vec_ref[index_vec[i]]);
        score_vec_ordered.push_back(score_ref[index_vec[i]]);
    }

    //assign pointers to look at new ordered vectors
    *vec = vec_ordered;
    *score_vec = score_vec_ordered;
}

//calculates intersection of two rectangles divided by the union of two rectangles
float calculate_IOU(cv::Rect a, cv::Rect b)
{
    //get area of a and its bottom right co-ordinates
    float areaA = a.area();

    float areaA_br_x = a.br().x;
    float areaA_br_y = a.br().y;

    //if its area is zero return 0
    if (areaA <= 0.0)
    {
        return 0.0;
    }

    //same for rectangle b
    float areaB = b.area();

    if (areaB <= 0.0)
    {
        return 0.0;
    }
    float areaB_br_x = b.br().x;
    float areaB_br_y = b.br().y;

    //calculate intersection by getting the tl and br points inside the intersection
    float intersection_left_x = std::max(a.tl().x, b.tl().x);
    float intersection_left_y = std::max(a.tl().y, b.tl().y);
    float intersection_bottom_x = std::min(areaA_br_x, areaB_br_x);
    float intersection_bottom_y = std::min(areaA_br_y, areaB_br_y);

    //the calculate the width and height and calculate area from that
    float intersection_width = std::max(intersection_bottom_x - intersection_left_x, (float)0.0);
    float intersection_height = std::max(intersection_bottom_y - intersection_left_y, (float)0.0);

    float intersection_area = intersection_width * intersection_height;

    //return intersection area divided by union area - intersection area
    return (float) intersection_area / (float) (areaA + areaB - intersection_area);
}

//calculate non-maximum suppression based on a threshold across a vector list of boxes and a vector list of confidence scores
//place maximum boxes for an area into a new list called selected.
void nms(std::vector<cv::Rect>* boxes, std::vector<float>* scores, std::vector<cv::Rect>* selected, float thresh)
{

    //sort the boxes so most confident boxes are first
    sortParallelVector(boxes, scores);

    //create a reference to it so it can be easily indexed
    std::vector<cv::Rect>& boxes_ref = *boxes;
    //std::vector<float>& score_ref = *scores;

    //create a new vector list called active which has a corresponding value for each box
    //initially each box is considered active so true is pushed back for every position
    std::vector<bool> active;

    for (std::size_t i = 0; i != boxes->size(); i++)
    {
        active.push_back(true);
    }

    //get number of still active boxes (which is just total size at this point)
    int num_active = active.size();

    //set up a stop variable
    bool done = false;

    //for each box...
    for (std::size_t i = 0; i != boxes->size(); i++)
    {
        //check if it's active first
        if (active[i])
        {
            //if it is then push this box into our selected boxes list (it should be
            //the most confident box in a given area)
            cv::Rect box_a = boxes_ref[i];
            selected->push_back(box_a);

            //for every other box after this one
            for (std::size_t j = i+1; j != boxes->size(); j++)
            {
                //if this box is still active
                if (active[j])
                {
                    //calculate it's IOU with outer loop box
                    cv::Rect box_b = boxes_ref[j];

                    float iou = calculate_IOU(box_a, box_b);

                    //if it has too big an IOU then this box should be marked as inactive
                    if (iou > thresh)
                    {
                        active[j] = false;
                        num_active--;

                        //if there are no more active boxes we are done
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

//this function takes in the original image as well as a rectangle box for the region we want
//to binarise
//it outputs a smaller image of the region of interest that has been binarised
cv::Mat binariseBox(cv::Mat img, cv::Rect inBox)
{
    //convert to gray and smooth
    cv::Mat grayImg;
    cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    cv::Mat boxImg(grayImg, inBox);

    //smooth the box - possibly remove?
    cv::Mat boxSmoothed;
    cv::medianBlur(boxImg, boxSmoothed, 5);

    //perform otsu's thresholding
    cv::Mat threshBox;

    cv::threshold(boxSmoothed, threshBox, 0, 255, cv::THRESH_OTSU);
    boxImg.release();
    boxSmoothed.release();
    grayImg.release();

    //if two more or more corners are black we should invert the image (i.e. the background has
    //been made black in thresholding).
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

    //convert thresholded image to BGR format so it has 3 channels all of the binary values
    cv::Mat threshBGR;
    cvtColor(threshBox, threshBGR, cv::COLOR_GRAY2BGR);

    return threshBGR;
}

//perform resizing and center cropping in line with mobilenetv3 translations before the image
//goes into the model
cv::Mat translationPreProcess(cv::Mat* srcImg)
{
    cv::Mat binRoiR;
    cv::resize(*srcImg, binRoiR, cv::Size(232, 232));

    const int cropSize = 224;
    const int offsetW = (binRoiR.cols - cropSize) / 2;
    const int offsetH = (binRoiR.rows - cropSize) / 2;
    const cv::Rect roiBin(offsetW, offsetH, cropSize, cropSize);
    return binRoiR(roiBin).clone();
}


//this function takes in an input image and will assign values to the max and argmax variables
//which determine the confidence of the prediction as well as which class was predicted
void getTranslation(cv::Mat* srcImg, float* max, std::string* argMax)
{
    //convert input image into ncnn format
    ncnn::Mat input = ncnn::Mat::from_pixels(srcImg->data, ncnn::Mat::PIXEL_BGR2RGB, srcImg->cols,
                                             srcImg->rows);

    //normalise the values according to mobilenet translations
    float means[] = {0.485f*255.f, 0.456f*255.f, 0.406*255.f};
    float norms[] = {1/0.229f/255.f, 1/0.224/255.f, 1/0.225f/255.f};
    input.substract_mean_normalize(means, norms);

    //create extractor which takes in input and gives output
    // Inference
    ncnn::Extractor extractor = translationModel.create_extractor();
    //extractor.set_light_mode(true);
    extractor.input("input", input);
    ncnn::Mat output;
    extractor.extract("output", output);
    //output in this example is an array of confidence scores

    //step through each score to find the maximum score and class it belongs to
    *max = 0.0;
    for (int j = 0; j < output.w; j++) {
        if (output[j] > *max) {
            *max = output[j];
            *argMax = std::to_string(j+1);
        }
    }
}

//this takes in two images, the region where character has been translated and that same region
//on the image we are overwriting onto with the character overlay
void overlayTranslation(cv::Mat roi, cv::Mat replaceroi, float* max, std::string* argMax)
{
    //check that prediciton is above confidence threshold
    float confThreshold = 0.7;
    if (*max >= confThreshold)
    {
        //if it is then get character bitmap image
        std::string filename = "overlays/";
        filename.append(*argMax);
        filename.append(".bmp");

        // load file from assets
        AAsset *asset = AAssetManager_open(mgr, filename.c_str(), 0);
        long size = AAsset_getLength(asset);
        uchar *buffer = (uchar *) malloc(sizeof(uchar) * size);
        AAsset_read(asset, buffer, size);
        AAsset_close(asset);

        //convert bitmap image file into an rgb image
        cv::Mat rawData(1, size, CV_8UC1, (void *) buffer);
        cv::Mat decodedImage = imdecode(rawData, cv::IMREAD_COLOR);

        cv::Mat decodeColor;
        cvtColor(decodedImage, decodeColor, cv::COLOR_BGR2RGB);
        decodedImage.release();

        //resize our overlay to the region we are replacing size
        cv::Mat overlayImg;
        cv::resize(decodeColor, overlayImg, roi.size());
        decodeColor.release();

        //create an alpha mask to ensure our overlay has alpha values set to full, making it fully visible
        cv::Mat alphaMask(roi.rows, roi.cols, CV_8UC1, cv::Scalar(255));

        cv::cvtColor(overlayImg, overlayImg, cv::COLOR_RGB2RGBA);
        cv::cvtColor(roi, roi, cv::COLOR_BGR2RGBA);

        //overwrite alpha values of overlay with our mask
        std::vector<cv::Mat>channels(4);
        cv::split(overlayImg, channels);

        channels[3] = alphaMask;

        cv::merge(channels, overlayImg);

        //add the overlay image and original region in a weighted fashion to the replacement image region
        //overlay image has weight of 1, original region has weight of 0.
        addWeighted(overlayImg, 1, roi, 0, 0, replaceroi);


        //when we overwrite the replacement region (which was a section of the image we want to put the overlay onto),
        //it will propagate back to the original image it was taken from overwriting that region in itself as well

        overlayImg.release();
    }
}

//assign asset manager variable as the asset manager that got passed
void assignManager(AAssetManager* manager){
    mgr = manager;
}

//this function is used for live mode capture and photo mode capture, it steps through all the above
//functions in a pipeline fashion in order to produce an output image with translated characters overlaid
cv::Mat captureImage(AAssetManager* manager, cv::Mat srcImg, int option) {
    auto beg = std::chrono::high_resolution_clock::now();
    assignManager(manager);

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

    //option parameter is used here so if the app is in live mode, the image we overlay the characters
    //onto should be a blank and completely invisible image (alpha channel = 0)
    //Using this alpha trick we can overlay a photo onto the user's view without obscuring the live
    //camera view behind it and make it appear like translations are appearing in realtime
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

        overlayTranslation(roi, replaceroi, &max_class, &argMax_class);

    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "total time %f", duration.count()/1000.0);
    return replaceImg;
}

//this function is used for draw mode capture, it steps through some of the above functions in
//a pipeline fashion to translate the bounding box provided by the rectangle the user drew in draw
//mode and produce an image where there is a character overlaid onto that region
cv::Mat captureBoxImage(AAssetManager* manager, cv::Mat srcImg, int x, int y, int w, int h) {
    auto beg = std::chrono::high_resolution_clock::now();
    mgr = manager;
    cv::Rect box = cv::Rect(x, y, w, h);

    //cv::Mat img;
    //cvtColor(srcImg, img, cv::COLOR_RGBA2BGR);
    float max_class;
    std::string argMax_class;
    cv::Mat roi = srcImg(box);
    cv::Mat replaceroi = srcImg(box);
    cv::Mat binRoi = binariseBox(srcImg, box);
    binRoi = translationPreProcess(&binRoi);
    getTranslation(&binRoi, &max_class, &argMax_class);
    binRoi = translationPreProcess(&binRoi);

    overlayTranslation(roi, replaceroi, &max_class, &argMax_class);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - beg);
    __android_log_print(ANDROID_LOG_DEBUG, "WallClock", "total time %f", duration.count()/1000.0);
    return srcImg;
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
