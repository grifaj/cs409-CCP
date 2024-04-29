#include <jni.h>
#include <string>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <android/asset_manager.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "enhance.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/core/types_c.h>

#include "net.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_stringFromJNI(JNIEnv *env, jobject thiz) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_validate(JNIEnv *env, jobject thiz, jlong mad_addr_gr,jlong mat_addr_rgba) {
    cv::Rect();
    cv::Mat();
    ncnn::Mat();
    std::string hello2="libraries load";
    return env->NewStringUTF(hello2.c_str());
}
extern "C"
JNIEXPORT void JNICALL
Java_com_android_example_cpp_1test_MainActivity_preloadModels(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    preloadModels(mgr);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_android_example_cpp_1test_CameraActivity_callBoundingBoxes(JNIEnv *env, jobject thiz, jlong image, jobject assetManager, jint opt) {
    cv::Mat* matImage=(cv::Mat*)image;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    *matImage =  captureImage(mgr,*matImage, (int)opt);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_android_example_cpp_1test_CameraActivity_callBoundingBoxes2(JNIEnv *env, jobject thiz, jlong image, jint x_box, jint y_box, jint w_box, jint h_box,  jobject assetManager) {
    cv::Mat* matImage=(cv::Mat*)image;
    int x = x_box;
    int y = y_box;
    int w = w_box;
    int h = h_box;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    *matImage =  captureBoxImage(mgr,*matImage,x,y,w,h);
}

// testing functions
extern "C"
JNIEXPORT jstring JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_testModelsLoad(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    std::string pass =  "libraries load";
    std::string fail =  "fail";
    ncnn::Net translationModel;
    ncnn::Net detectionModel;
    int ret = translationModel.load_param(mgr,"mobilenet_v3_large-sim-opt.param");
    if (ret) {
        return env->NewStringUTF(fail.c_str());
    }
    ret = translationModel.load_model(mgr, "mobilenet_v3_large-sim-opt.bin");
    if (ret) {
        return env->NewStringUTF(fail.c_str());
    }
    ret = detectionModel.load_param(mgr,"model.ncnn.param");
    if (ret) {
        return env->NewStringUTF(fail.c_str());
    }
    ret = detectionModel.load_model(mgr, "model.ncnn.bin");
    if (ret) {
        return env->NewStringUTF(fail.c_str());
    }
    return env->NewStringUTF(pass.c_str());
}
extern "C"
JNIEXPORT jfloat JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_calculate_1IOU(JNIEnv *env, jobject thiz, jint flag) {
    int flag_num = (int) flag;
    float ret = 0.0;

    if (flag == 1)
    {
        cv::Rect a = cv::Rect(0, 0, 100, 100);
        ret = calculate_IOU(a, a);
    } else if (flag == 2){
        cv::Rect a = cv::Rect(0, 0, 100, 100);
        cv::Rect b = cv::Rect(200, 200, 100, 100);
        ret = calculate_IOU(a, b);
    }
    return ret;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_greyImage(JNIEnv *env, jobject thiz) {
    cv::Mat srcImg(320, 240, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat output = grayImage(&srcImg);
    return  output.channels() == 1;

}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_binariseImg(JNIEnv *env, jobject thiz,
                                                                       jlong image) {
    cv::Mat* matImage=(cv::Mat*)image;
    cv::Rect inBox(0,0,20,20);
    cv::Mat bin = binariseBox(*matImage, inBox);

    // check all values 255 or 0
    uint8_t* pixelPtr = (uint8_t*)bin.data;
    bool binary = true;
    for (int i = 0; i < bin.size().height ; ++i) {
        for (int j = 0; j < bin.size().width; ++j) {
            if (pixelPtr[i*bin.cols + j] != 0 && pixelPtr[i*bin.cols + j] != 255){
                binary = false;
                break;
            }
        }
    }
    return bin.channels() == 3 && bin.size().height == 20 && bin.size().width == 20 && binary;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_JNI_1sortParallelVector(JNIEnv *env,
                                                                                   jobject thiz) {
    std::vector<cv::Rect>* vec = new std::vector<cv::Rect>();
    vec->push_back(cv::Rect(0,0,10,10));
    vec->push_back(cv::Rect(15,100,97,88));
    vec->push_back(cv::Rect(10,20,30,40));

    std::vector<float> *score_vec =  new std::vector<float>();
    score_vec->push_back(10);
    score_vec->push_back(30);
    score_vec->push_back(100);

    sortParallelVector(vec, score_vec);

    bool rects = false;
    if(vec[0][0] == cv::Rect(10,20,30,40) &&  vec[0][1] == cv::Rect(15,100,97,88) && vec[0][2] == cv::Rect(0,0,10,10))rects = true;
    bool scores = false;
    if(score_vec[0][0] == 100.0 && score_vec[0][1] == 30 && score_vec[0][2] == 10) scores = true;

    return rects && scores;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_JNI_1non_1max_1suppression(JNIEnv *env,
                                                                                      jobject thiz) {

    std::vector<cv::Rect>* boxes = new std::vector<cv::Rect>();
    boxes->push_back(cv::Rect(100,100,50,50));
    boxes->push_back(cv::Rect(90,100,50,50));
    boxes->push_back(cv::Rect(10,20,30,40));
    boxes->push_back(cv::Rect(5,20,30,40));

    std::vector<float> *scores =  new std::vector<float>();
    scores->push_back(100);
    scores->push_back(1);
    scores->push_back(100);
    scores->push_back(1);

    std::vector<cv::Rect>* selected = new std::vector<cv::Rect>();

    // should only keep 2 boxes
    nms(boxes, scores, selected, 0.5);

    bool output = (selected[0][0] == cv::Rect(100,100,50,50) && selected[0][1] == cv::Rect(10,20,30,40));
    return output && selected->size() == 2;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_padImage(JNIEnv *env, jobject thiz) {
    cv::Mat srcImg(512, 240, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat output = padImage(&srcImg);
    return output.size().height == 512 && output.size().width == 512;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_scale_1img(JNIEnv *env, jobject thiz) {
    cv::Mat srcImg(1024, 2048, CV_8UC3, cv::Scalar(100, 100, 100));
    float sf;

    cv::Mat output =  resizeSF(&srcImg, &sf);

    return output.size().height == 256 && output.size().width ==  512 && sf == 0.25;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_test_1detection(JNIEnv *env,jobject thiz, jlong image, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    cv::Mat* srcImg=(cv::Mat*)image;

    cv::Mat grayImg = grayImage(srcImg);

    cv::Mat graySmoothed = preProcessImage(&grayImg);
    grayImg.release();

    cv::Mat grayBGR = grayToBGR(&graySmoothed);
    graySmoothed.release();

    float sf;
    cv::Mat sfScaled = resizeSF(&grayBGR, &sf);
    grayBGR.release();

    cv::Mat imPad = padImage(&sfScaled);
    sfScaled.release();

    assignManager(mgr);
    loadDetectionModel();

    std::vector<cv::Rect>* boxes = new std::vector<cv::Rect>();
    std::vector<float>* confidences = new std::vector<float>();

    detectModel(&imPad, srcImg, &sf, boxes, confidences);

    return boxes->size() >= 0 && confidences->size() >= 0;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_translation_1Pre(JNIEnv *env,jobject thiz) {
    cv::Mat srcImg(512, 240, CV_8UC3, cv::Scalar(100, 100, 100));

    cv::Mat output = translationPreProcess(&srcImg);

    return output.rows == 224 && output.cols == 224;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_get_1translation(JNIEnv *env, jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    assignManager(mgr);
    cv::Mat srcImg(512, 240, CV_8UC3, cv::Scalar(100, 100, 100));
    float max;
    std::string argMax;

    loadTranslationModel();
    getTranslation(&srcImg, &max, &argMax);

    int argMax_int = std::stoi(argMax);

    return max < 0.7 && argMax_int >= 1 && argMax_int <= 1000;
}
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_test_1overlay(JNIEnv *env,jobject thiz, jobject assetManager) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    assignManager(mgr);

    cv::Mat roi(512, 240, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat replaceroi(512, 240, CV_8UC3, cv::Scalar(100, 100, 100));

    float max = 0.8;
    std::string argMax = "1";

    overlayTranslation(roi, replaceroi, &max, &argMax);

    return  roi.data != replaceroi.data;
}