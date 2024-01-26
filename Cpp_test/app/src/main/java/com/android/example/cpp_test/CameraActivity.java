package com.android.example.cpp_test;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.concurrent.ExecutionException;

public class CameraActivity extends AppCompatActivity {
    static {
        System.loadLibrary("cpp_test");
    }
    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageView photoPreview;
    private ImageView closePhotoPreview;
    private CameraSelector lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
    Mat cvMat;
    Bitmap bitmapPhoto;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        ImageView cameraShutter = findViewById(R.id.cameraShutter);
        photoPreview = findViewById(R.id.photoPreview);
        ImageView switchLens = findViewById(R.id.switchLens);
        closePhotoPreview = findViewById(R.id.closePhotoPreview);

        // create camera view
        showImagePreview();

        //cameraProviderFuture.addListener(this::showImagePreview, ContextCompat.getMainExecutor(this));

        // freeze camera preview on taken photo
        cameraShutter.setOnClickListener(v -> {
            // take photo from preview
            bitmapPhoto = previewView.getBitmap();
            detectChars(); // sets bitmap to have chars on it

            // set photoPreview to image and make it visible
            photoPreview.setImageBitmap(bitmapPhoto);
            photoPreview.setVisibility(View.VISIBLE);

            // also make preview exit button visible
            closePhotoPreview.setVisibility(View.VISIBLE);
        });

        // return to camera preview
        closePhotoPreview.setOnClickListener(v -> {
            // set components back to invisible
            photoPreview.setVisibility(View.GONE);
            closePhotoPreview.setVisibility(View.GONE);
        });

        switchLens.setOnClickListener(v -> {
            if (lensFacing == CameraSelector.DEFAULT_FRONT_CAMERA) lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
            else if (lensFacing == CameraSelector.DEFAULT_BACK_CAMERA) lensFacing = CameraSelector.DEFAULT_FRONT_CAMERA;
            showImagePreview();
        });
    }

    private void showImagePreview() {
        cameraProviderFuture.addListener(() -> {
            ImageCapture imageCapture = new ImageCapture.Builder()
                    .setTargetRotation(previewView.getDisplay().getRotation())
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build();

            try {
                ProcessCameraProvider processCameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                processCameraProvider.unbindAll();

                // lensFacing is used here
                processCameraProvider.bindToLifecycle((LifecycleOwner)this, lensFacing, imageCapture, preview);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void detectChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes(cvMat.getNativeObjAddr());
        //convert back
        Utils.matToBitmap(cvMat,bitmapPhoto);
    }

    public native void callBoundingBoxes(long image);
}
