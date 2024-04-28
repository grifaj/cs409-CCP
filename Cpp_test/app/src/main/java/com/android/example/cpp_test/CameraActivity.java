package com.android.example.cpp_test;

import android.animation.Animator;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.DisplayOrientedMeteringPointFactory;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.FocusMeteringAction;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.MeteringPoint;
import androidx.camera.core.MeteringPointFactory;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.util.Objects;
import java.util.concurrent.ExecutionException;

import dalvik.annotation.optimization.FastNative;

public class CameraActivity extends AppCompatActivity implements SensorEventListener {
    static {
        System.loadLibrary("cpp_test");
    }
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Camera camera;
    private CameraSelector lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
    private ProcessCameraProvider processCameraProvider;

    private PreviewView previewView;
    private ImageView photoPreview, closePhotoPreview, swapImage, switchLens, liveMode, drawMode, cameraShutter;
    private DrawView drawView;
    private View resetZoom;


    private Mat cvMat;
    private Bitmap bitmapPhoto, originalBitmap;

    private SensorManager sensorMan;
    private Sensor accelerometer;

    protected float zoom;
    private float mAccel;
    private float mAccelCurrent;

    private int accelThreshCount;

    private boolean predicted = false;
    private boolean translate = true;
    private boolean drawingMode;
    private boolean videoMode;
    private boolean previewMode;

    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        drawingMode = false;
        videoMode = false;
        previewMode = false;

        previewView = findViewById(R.id.previewView);
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraShutter = findViewById(R.id.cameraShutter);
        drawMode = findViewById(R.id.drawMode);
        liveMode = findViewById(R.id.liveMode);
        photoPreview = findViewById(R.id.photoPreview);
        switchLens = findViewById(R.id.switchLens);
        closePhotoPreview = findViewById(R.id.closePhotoPreview);
        resetZoom = findViewById(R.id.resetZoom);
        swapImage = findViewById(R.id.swapImage);
        drawView = findViewById(R.id.drawView);

        sensorMan = (SensorManager)getSystemService(SENSOR_SERVICE);
        accelerometer = sensorMan.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorMan.unregisterListener(this, accelerometer);

        // create camera view
        showImagePreview();

        // freeze camera preview on taken photo
        cameraShutter.setOnClickListener(v -> {
            // take photo from preview
            originalBitmap = previewView.getBitmap();
            assert originalBitmap != null;
            bitmapPhoto = originalBitmap.copy(originalBitmap.getConfig(), true);
            previewMode = true;
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
            switchLens.setVisibility(View.GONE);
            resetZoom.setVisibility(View.GONE);
            cameraShutter.setVisibility(View.GONE);
            drawMode.setVisibility(View.GONE);
            liveMode.setVisibility(View.GONE);

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
                if (videoMode)
                {
                    videoMode = false;
                    liveMode.setBackgroundResource(R.drawable.circle_background);
                    photoPreview.setVisibility(View.GONE);
                    accelThreshCount = 0;
                    sensorMan.unregisterListener(this, accelerometer);
                }

                drawingMode = true;
                drawMode.setBackgroundResource(R.drawable.pressed_background);

                drawView.setVisibility(View.VISIBLE);
            }

            drawView.setDrawMode(drawingMode);
        });

        liveMode.setOnClickListener(v -> {
            if (videoMode)
            {
                videoMode = false;
                liveMode.setBackgroundResource(R.drawable.circle_background);
                photoPreview.setVisibility(View.GONE);
                accelThreshCount = 0;
                sensorMan.unregisterListener(this, accelerometer);
            }
            else
            {
                if (drawingMode)
                {
                    drawingMode = false;
                    drawMode.setBackgroundResource(R.drawable.circle_background);
                    drawView.setVisibility(View.GONE);
                    drawView.setDrawMode(drawingMode);
                }
                videoMode = true;
                liveMode.setBackgroundResource(R.drawable.pressed_background);

                sensorMan.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI);
            }
        });

        resetZoom.setOnClickListener(v -> {
            resetZoom.setAlpha(0.5f); // dim to animate
            camera.getCameraControl().setZoomRatio(1); // set zoom back to normal
            zoom = 1.0f;
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal
        });

        // return to camera preview
        closePhotoPreview.setOnClickListener(v -> {
            closePreview();
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
//                Log.d("TOUCH", "Passing to next listener");
                return false;
            }
//            Log.d("TOUCH", "Screen touched");
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
//            Log.d("TOUCH", "Passed to me");
            return drawView.onTouchEvent(event);
        });

    }

    public void closePreview(){
        showImagePreview();
        // set components back to invisible
        photoPreview.setVisibility(View.GONE);
        closePhotoPreview.setVisibility(View.GONE);
        swapImage.setVisibility(View.GONE);

        // add other buttons
        switchLens.setVisibility(View.VISIBLE);
        cameraShutter.setVisibility(View.VISIBLE);
        drawMode.setVisibility(View.VISIBLE);
        liveMode.setVisibility(View.VISIBLE);
        resetZoom.setVisibility(View.VISIBLE);
        previewMode = false;
    }

    @Override
    public void onBackPressed() {
        Log.d("backButton", "Pressed");
        if (previewMode) {
            closePreview();
            return;
        }
        moveTaskToBack(true);

        // don't judge, it works
        if(1==0){
            super.onBackPressed();
        }

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (videoMode)
        {
            Log.d("VIDEO", "Detected sensor change");
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER)
            {
                float[] mGravity = event.values.clone();
                // Shake detection
                float x = mGravity[0];
                float y = mGravity[1];
                float z = mGravity[2];
                float mAccelLast = mAccelCurrent;
                mAccelCurrent = (float)Math.sqrt(x*x + y*y + z*z);
                float delta = mAccelCurrent - mAccelLast;
                mAccel = mAccel * 0.9f + delta;

                if (Math.abs(mAccel) < 0.1)
                {
                    accelThreshCount +=1;
                    if (accelThreshCount % 30 == 0)
                    {
                        predicted = false;
                    }
                }
                else
                {
                    accelThreshCount = 0;
                    predicted = false;
                }
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();
        sensorMan.registerListener(this, accelerometer,
                SensorManager.SENSOR_DELAY_UI);
    }

    @Override
    protected void onPause()
    {
        super.onPause();
        sensorMan.unregisterListener(this);
    }

    @OptIn(markerClass = ExperimentalGetImage.class) private void showImagePreview()
    {

        cameraProviderFuture.addListener(() -> {
            ImageAnalysis imageAnalysis =
                    new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();

            imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), imageProxy -> {

                //Log.d("acc", String.valueOf(mAccel));
                if (accelThreshCount >= 5 && !predicted) {
                    Image image = imageProxy.getImage();
                    assert image != null;
                    bitmapPhoto = previewView.getBitmap();

                    photoPreview.setImageBitmap(bitmapPhoto);
                    photoPreview.setVisibility(View.VISIBLE);
                    detectMovingChars();
                    predicted = true;

                } else if (accelThreshCount >= 5) {
                    assert true;
                } else {
                    photoPreview.setVisibility(View.GONE);
                }
                // after done, release the ImageProxy object
                imageProxy.close();
            });


            try {
                processCameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                processCameraProvider.unbindAll();

                // lensFacing is used here
                camera = processCameraProvider.bindToLifecycle(this, lensFacing, imageAnalysis, preview);
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
                    }
                    @Override
                    public void onAnimationCancel(@NonNull Animator animation)
                    {
                    }
                    @Override
                    public void onAnimationRepeat(@NonNull Animator animation)
                    {
                    }
                });
    }
    private void detectChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes(cvMat.getNativeObjAddr(), getAssets(), 0);
        //convert back
        Utils.matToBitmap(cvMat,bitmapPhoto);
    }

    private void detectMovingChars() {
        cvMat = new Mat();
        // convert to opencv Matrix format
        Utils.bitmapToMat(bitmapPhoto, cvMat);
        callBoundingBoxes(cvMat.getNativeObjAddr(), getAssets(), 1);
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
            accelThreshCount = 0;
            float currentZoomRatio = Objects.requireNonNull(camera.getCameraInfo().getZoomState().getValue()).getZoomRatio();
            float delta = detector.getScaleFactor();
            camera.getCameraControl().setZoomRatio(currentZoomRatio * delta);
            zoom = currentZoomRatio * delta;
            return true;
        }
    }

    @FastNative
    public native void callBoundingBoxes(long image, AssetManager assetManager, int opt);
    @FastNative
    public native void callBoundingBoxes2(long image, int x, int y, int w, int h, AssetManager assetManager);
}
