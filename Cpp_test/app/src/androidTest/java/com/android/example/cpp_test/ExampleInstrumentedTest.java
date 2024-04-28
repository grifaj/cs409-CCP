package com.android.example.cpp_test;

import android.content.Context;
import android.content.res.AssetManager;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.core.Rect;

import static org.junit.Assert.*;

import java.io.IOException;
import java.io.InputStream;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class ExampleInstrumentedTest {
    static {
        System.loadLibrary("cpp_test");
    }
    private final Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();
    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("com.android.example.cpp_test", appContext.getPackageName());
    }
    @Test
    public void jni_isLinked(){
        assertEquals("Hello from C++",stringFromJNI());
    }
    @Test
    public void libraries_load(){
        assertEquals("libraries load", validate(500,500));
    }
    @Test
    public void test_ReadAssets() throws IOException {
        InputStream input = ctx.getAssets().open("overlays/1.bmp");
        Assert.assertNotNull(input);
    }
    @Test
    public void models_load(){assertEquals("libraries load", testModelsLoad(ctx.getAssets()));}
    @Test
    public void IoU_full(){
        Rect a = new Rect(0, 0, 100, 100);
        double delta = 0.01;
        assertEquals(1,calculate_IOU(a, a),delta);

    }
    // convert image to greyscale correctly
    // padded yolo image properly
    public native String validate(long madAddrGr,long matAddrRgba);
    public native String stringFromJNI();
    public native String testModelsLoad(AssetManager assetManager);
    public native float calculate_IOU(Rect a, Rect b);
}