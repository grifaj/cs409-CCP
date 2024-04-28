package com.android.example.cpp_test;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.test.core.app.ApplicationProvider;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
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
    public void IoU_full(){assertEquals(1,calculate_IOU(1),0);}
    @Test
    public void IoU_empty(){assertEquals(0,calculate_IOU(2),0);}
    @Test
    public void convert2Grey(){assertTrue(greyImage());}
    @Test
    public void binariseImg() throws IOException {
        InputStream stream = ctx.getAssets().open("seal-script-test.jpg");

        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bmp = BitmapFactory.decodeStream(stream, null, bmpFactoryOptions);
        Mat ImageMat = new Mat();
        Utils.bitmapToMat(bmp, ImageMat);

        assertTrue(binariseImg(ImageMat.getNativeObjAddr()));
    }
    @Test
    public void sortParallelVector(){assertTrue(JNI_sortParallelVector());}
    @Test
    public void non_max_suppression(){assertTrue(JNI_non_max_suppression());}
    @Test
    public void image_pad(){assertTrue(padImage());}

    public native String validate(long madAddrGr,long matAddrRgba);
    public native String stringFromJNI();
    public native String testModelsLoad(AssetManager assetManager);
    public native float calculate_IOU(int flag);
    public native boolean greyImage();
    public native boolean binariseImg(long image);
    public native boolean JNI_sortParallelVector();
    public native boolean JNI_non_max_suppression();
    public native boolean padImage();
}