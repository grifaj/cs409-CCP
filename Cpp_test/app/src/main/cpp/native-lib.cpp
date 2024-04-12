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

#include "net.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_android_example_cpp_1test_MainActivity_stringFromJNI(JNIEnv *env, jobject thiz) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
extern "C"
JNIEXPORT jstring JNICALL
Java_com_android_example_cpp_1test_MainActivity_validate(JNIEnv *env, jobject thiz, jlong mad_addr_gr,jlong mat_addr_rgba) {
    cv::Rect();
    cv::Mat();
    std::string hello2="hello from validate";
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
Java_com_android_example_cpp_1test_CameraActivity_callBoundingBoxes(JNIEnv *env, jobject thiz, jlong image, jobject assetManager) {
    cv::Mat* matImage=(cv::Mat*)image;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    *matImage =  captureImage(mgr,*matImage);
}
