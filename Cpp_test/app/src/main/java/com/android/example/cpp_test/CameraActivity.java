package com.android.example.cpp_test;

import android.animation.Animator;
import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.NonNull;
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
import java.util.ArrayList;
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
    private ImageView photoPreview, closePhotoPreview, swapImage;
    private View resetZoom;
    private CameraSelector lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
    private DrawView drawView;
    private Mat cvMat;
    private Bitmap bitmapPhoto, originalBitmap;
    private ProcessCameraProvider processCameraProvider;
    private boolean translate = true;
    protected float zoom;

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
        swapImage = findViewById(R.id.swapImage);
        drawView = findViewById(R.id.drawView);

        // create camera view
        showImagePreview();

        //cameraProviderFuture.addListener(this::showImagePreview, ContextCompat.getMainExecutor(this));

        // freeze camera preview on taken photo
        cameraShutter.setOnClickListener(v -> {
            // take photo from preview
            originalBitmap = previewView.getBitmap();
            assert originalBitmap != null;
            bitmapPhoto = originalBitmap.copy(originalBitmap.getConfig(), true);
            if (!drawingMode)
            {
                detectChars(); // sets bitmap to have chars on it
            }
            else
            {
                detectBoxChars();
            }

            // set photoPreview to image and make it visible
            photoPreview.setImageBitmap(bitmapPhoto);
            photoPreview.setVisibility(View.VISIBLE);

            // remove other buttons
            resetZoom.setVisibility(View.GONE);

            // stop camera view
            processCameraProvider.unbindAll();

            // also make preview exit button visible
            closePhotoPreview.setVisibility(View.VISIBLE);
            swapImage.setVisibility(View.VISIBLE);
        });

        swapImage.setOnClickListener(v -> {
            resetZoom.setAlpha(0.5f); // dim to animate
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal

            // swap to blank image
            if(translate){
                photoPreview.setImageBitmap(originalBitmap);
            }else{
                // swap to translation overlay
                photoPreview.setImageBitmap(bitmapPhoto);
            }
            translate = !translate;

        });

        drawMode.setOnClickListener(v -> {
            if (drawingMode)
            {
                drawingMode = false;
                drawMode.setBackgroundResource(R.drawable.circle_background);

                drawView.setVisibility(View.GONE);
            }
            else
            {
                drawingMode = true;
                drawMode.setBackgroundResource(R.drawable.pressed_background);

                drawView.setVisibility(View.VISIBLE);
            }

            drawView.setDrawMode(drawingMode);
        });

        resetZoom.setOnClickListener(v -> {
            resetZoom.setAlpha(0.5f); // dim to animate
            camera.getCameraControl().setZoomRatio(1); // set zoom back to normal
            zoom = 1.0f;
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal
        });

        // return to camera preview
        closePhotoPreview.setOnClickListener(v -> {
            showImagePreview();
            // set components back to invisible
            photoPreview.setVisibility(View.GONE);
            closePhotoPreview.setVisibility(View.GONE);
            swapImage.setVisibility(View.GONE);

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
            if (drawingMode)
            {
                Log.d("TOUCH", "Passing to next listener");
                return false;
            }
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

                    animateFocusRing(x, y);

                } catch (Exception CameraInfoUnavailableException)
                {
                    //Log.d("ERROR", "Cannot Access Camera", CameraInfoUnavailableException);
                    Log.d("ERROR", "Cannot Access Camera");
                }

                return true;
            }
            return false;
        });

        drawView.setOnTouchListener((v, event) -> {
            Log.d("TOUCH", "Passed to me");
            return drawView.onTouchEvent(event);
        });

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
                camera.getCameraControl().setZoomRatio(zoom);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void animateFocusRing(float x, float y)
    {
        ImageView focusRing = findViewById(R.id.focusRingView);

        float width = focusRing.getWidth();
        float height = focusRing.getHeight();

        focusRing.setX(x - width/2);
        focusRing.setY(y - height/2);

        focusRing.setVisibility(View.VISIBLE);
        focusRing.setAlpha(1F);

        focusRing.animate()
                .setStartDelay(500)
                .setDuration(300)
                .alpha(0F)
                .setListener(new Animator.AnimatorListener()
                {

                    @Override
                    public void onAnimationEnd(@NonNull Animator animation)
                    {
                        focusRing.setVisibility(View.INVISIBLE);

                    }
                    @Override
                    public void onAnimationStart(@NonNull Animator animation)
                    {
                        return;
                    }
                    @Override
                    public void onAnimationCancel(@NonNull Animator animation)
                    {
                        return;
                    }
                    @Override
                    public void onAnimationRepeat(@NonNull Animator animation)
                    {
                        return;
                    }
                });
    }

    private void detectChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes(cvMat.getNativeObjAddr(), getAssets());
        //convert back
        Utils.matToBitmap(cvMat,bitmapPhoto);
    }

    private void detectBoxChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes2(cvMat.getNativeObjAddr(), drawView.getTopX(), drawView.getTopY(), drawView.getBoxWidth(), drawView.getBoxHeight(), getAssets());
        //convert back
        Utils.matToBitmap(cvMat,bitmapPhoto);
    }

    public Display getDefaultDisplay(CameraActivity activity) {
        WindowManager windowManager = (WindowManager) activity.getSystemService(Context.WINDOW_SERVICE);
        return windowManager.getDefaultDisplay();
    }

    private class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener
    {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            float currentZoomRatio = Objects.requireNonNull(camera.getCameraInfo().getZoomState().getValue()).getZoomRatio();
            float delta = detector.getScaleFactor();
            camera.getCameraControl().setZoomRatio(currentZoomRatio * delta);
            zoom = currentZoomRatio * delta;
            return true;
        }
    }

    @FastNative
    public native void callBoundingBoxes(long image, AssetManager assetManager);
    @FastNative
    public native void callBoundingBoxes2(long image, int x, int y, int w, int h, AssetManager assetManager);
}