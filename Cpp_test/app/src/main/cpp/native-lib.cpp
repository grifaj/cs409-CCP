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
Java_com_android_example_cpp_1test_ExampleInstrumentedTest_calculate_1IOU(JNIEnv *env, jobject thiz, CvRect a, CvRect b) {
    float ret = calculate_IOU(a, b);
    return ret;
}