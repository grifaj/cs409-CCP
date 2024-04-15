package com.android.example.cpp_test;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import android.content.Context;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.DisplayOrientedMeteringPointFactory;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import android.content.res.AssetManager;
import android.view.Display;
import android.view.WindowManager;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.Objects;
import java.util.concurrent.ExecutionException;

import dalvik.annotation.optimization.FastNative;

public class CameraActivity extends AppCompatActivity {
    static {
        System.loadLibrary("cpp_test");
    }
    private boolean drawingMode;
    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Camera camera;
    private ImageView photoPreview;
    private ImageView closePhotoPreview;
    private View resetZoom;
    private CameraSelector lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
    Mat cvMat;
    Bitmap bitmapPhoto;
    ProcessCameraProvider processCameraProvider;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        drawingMode = false;

        previewView = findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        ImageView cameraShutter = findViewById(R.id.cameraShutter);
        ImageView drawMode = findViewById(R.id.drawMode);
        photoPreview = findViewById(R.id.photoPreview);
        ImageView switchLens = findViewById(R.id.switchLens);
        closePhotoPreview = findViewById(R.id.closePhotoPreview);
        resetZoom = findViewById(R.id.resetZoom);

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

            // remove other buttons
            resetZoom.setVisibility(View.GONE);

            // stop camera view
            processCameraProvider.unbindAll();

            // also make preview exit button visible
            closePhotoPreview.setVisibility(View.VISIBLE);
        });

        drawMode.setOnClickListener(v -> {
            if (drawingMode)
            {
                drawingMode = false;
                drawMode.setBackgroundResource(R.drawable.circle_background);
            }
            else
            {
                drawingMode = true;
                drawMode.setBackgroundResource(R.drawable.pressed_background);
            }
        });

        resetZoom.setOnClickListener(v -> {
            resetZoom.setAlpha(0.5f); // dim to animate
            camera.getCameraControl().setZoomRatio(1); // set zoom back to normal
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal
        });

        // return to camera preview
        closePhotoPreview.setOnClickListener(v -> {
            showImagePreview();
            // set components back to invisible
            photoPreview.setVisibility(View.GONE);
            closePhotoPreview.setVisibility(View.GONE);

            // add other buttons
            resetZoom.setVisibility(View.VISIBLE);
        });

        switchLens.setOnClickListener(v -> {
            if (lensFacing == CameraSelector.DEFAULT_FRONT_CAMERA) lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
            else if (lensFacing == CameraSelector.DEFAULT_BACK_CAMERA) lensFacing = CameraSelector.DEFAULT_FRONT_CAMERA;
            // spin icon
            switchLens.animate().rotation(180-switchLens.getRotation()).start();
            showImagePreview();
        });

        ScaleListener listener = new ScaleListener();
        ScaleGestureDetector scaleGestureDetector = new ScaleGestureDetector(previewView.getContext(), listener);

        previewView.setOnTouchListener((v, event) -> {
            Log.d("TOUCH", "Screen touched");
            scaleGestureDetector.onTouchEvent(event);
            if(event.getAction() == MotionEvent.ACTION_DOWN)
            {
                Log.d("TOUCH", "Pressed");
                return true;
            }
            if(event.getAction() == MotionEvent.ACTION_UP)
            {
                Log.d("TOUCH", "Released");
                //final Rect sensorArraySize = cameraProviderFuture.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                float x = event.getX();
                float y = event.getY();
                String output = "x = " + x + " y = " + y;
                Log.d("TOUCH", output);

                Display display = getDefaultDisplay(this);
                MeteringPointFactory factory  = new DisplayOrientedMeteringPointFactory(display, camera.getCameraInfo(), ((float) previewView.getWidth()), ((float) previewView.getHeight()));
                MeteringPoint meteringPoint = factory.createPoint(x,y);
                try
                {
                    FocusMeteringAction.Builder builder = new FocusMeteringAction.Builder(meteringPoint, FocusMeteringAction.FLAG_AF);
                    builder.disableAutoCancel();
                    camera.getCameraControl().startFocusAndMetering(builder.build());
                } catch (Exception CameraInfoUnavailableException)
                {
                    //Log.d("ERROR", "Cannot Access Camera", CameraInfoUnavailableException);
                    Log.d("ERROR", "Cannot Access Camera");
                }

                return true;
            }
            return false;
        });
    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener
    {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            float currentZoomRatio = Objects.requireNonNull(camera.getCameraInfo().getZoomState().getValue()).getZoomRatio();
            float delta = detector.getScaleFactor();
            camera.getCameraControl().setZoomRatio(currentZoomRatio * delta);
            return true;
        }
    }

    private void showImagePreview() {
        cameraProviderFuture.addListener(() -> {
            ImageCapture imageCapture = new ImageCapture.Builder()
                    .setTargetRotation(previewView.getDisplay().getRotation())
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build();

            try {
                processCameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                processCameraProvider.unbindAll();

                // lensFacing is used here
                camera = processCameraProvider.bindToLifecycle(this, lensFacing, imageCapture, preview);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void detectChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes(cvMat.getNativeObjAddr(), getAssets());
        //convert back
        Utils.matToBitmap(cvMat,bitmapPhoto);
    }

    public Display getDefaultDisplay(CameraActivity activity) {
        WindowManager windowManager = (WindowManager) activity.getSystemService(Context.WINDOW_SERVICE);
        return windowManager.getDefaultDisplay();
    }

    @FastNative
    public native void callBoundingBoxes(long image, AssetManager assetManager);
}
