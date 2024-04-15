package com.android.example.cpp_test;

import android.content.Context;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

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

    public native String validate(long madAddrGr,long matAddrRgba);
    public native String stringFromJNI();
}