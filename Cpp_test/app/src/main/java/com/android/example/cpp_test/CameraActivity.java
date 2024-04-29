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
    static
    {
        System.loadLibrary("cpp_test");
    }
    //set up different variables required for running and tracking system state.
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

    //override the super class on create method, this method will run when the class is first started
    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //set view on the phone to layout defined in activity main xml file
        setContentView(R.layout.activity_main);
        //phone should not be in any current mode.
        drawingMode = false;
        videoMode = false;
        previewMode = false;

        //get resources of xml such as icons so we can manipulate them later.
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

        //set up initial accelerometer sensor but unregister it for now
        sensorMan = (SensorManager)getSystemService(SENSOR_SERVICE);
        accelerometer = sensorMan.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorMan.unregisterListener(this, accelerometer);

        //call show image preview which will display the current camera feed
        showImagePreview();

        //set a listener for the take a photo button, when it is clicked it will...
        cameraShutter.setOnClickListener(v -> {
            // take photo from preview
            //get a bitmap photo of the current camera feed
            originalBitmap = previewView.getBitmap();
            assert originalBitmap != null;
            //duplicate it so we have original photo + translation photo
            bitmapPhoto = originalBitmap.copy(originalBitmap.getConfig(), true);
            //preview mode is now true -> i.e. they are viewing a photo
            previewMode = true;

            //if the user is not in draw mode when the photo was taken then just detect characters on
            //camera feed. otherwise translate character inside the drawing box
            if (!drawingMode)
            {
                detectChars();
            }
            else
            {
                detectBoxChars();
            }

            //set a hidden photo preview to be the bitmap photo we just took
            //and make it visible
            photoPreview.setImageBitmap(bitmapPhoto);
            photoPreview.setVisibility(View.VISIBLE);

            // remove user buttons so they can't interact with drawmode and live mode etc.
            switchLens.setVisibility(View.GONE);
            resetZoom.setVisibility(View.GONE);
            cameraShutter.setVisibility(View.GONE);
            drawMode.setVisibility(View.GONE);
            liveMode.setVisibility(View.GONE);

            //stop the camera feed from running while user is looking at the translated photo
            processCameraProvider.unbindAll();

            //make the close photo preview button visible so the user can stop looking at the photo
            //as well as the swap between translation and original image button
            closePhotoPreview.setVisibility(View.VISIBLE);
            swapImage.setVisibility(View.VISIBLE);
        });

        //set up a listener for the swap between original and translated image button to listen
        //for when it is clicked
        swapImage.setOnClickListener(v -> {
            //animate it to make it look fancy
            resetZoom.setAlpha(0.5f); // dim to animate
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal

            //swap photo preview to show the original image / translated image based on current state
            if(translate){
                photoPreview.setImageBitmap(originalBitmap);
            }else{
                // swap to translation overlay
                photoPreview.setImageBitmap(bitmapPhoto);
            }
            translate = !translate;

        });

        //set up a listener for the draw mode button
        drawMode.setOnClickListener(v -> {
            //if current in drawing mode then deactivate it, make it's view invisible
            //and set the button to appear deactivated.
            if (drawingMode)
            {
                drawingMode = false;
                drawMode.setBackgroundResource(R.drawable.circle_background);

                drawView.setVisibility(View.GONE);
            }
            else //otherwise...
            {
                //if live translation mode is active, then turn it off and unregister stuff to do with it
                //then set drawmode to true and make the draw view visibile so the user can interact with the box
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

            //alert the drawing view about its current state so it can either listen or not
            drawView.setDrawMode(drawingMode);
        });

        //set up listener for the live translation mode button when its clicked
        liveMode.setOnClickListener(v -> {
            //if live translation mode is active then deactivate it and unregister accelerometer listener
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
                //otherwise turn off drawing mode if it's currently on
                //and turn on live video mode and register accelerometer listener again
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

        //set a listener on the reset zoom button which will animate and set zoom back to 1x
        resetZoom.setOnClickListener(v -> {
            resetZoom.setAlpha(0.5f); // dim to animate
            camera.getCameraControl().setZoomRatio(1); // set zoom back to normal
            zoom = 1.0f;
            resetZoom.animate().alpha(1f).setDuration(1000); // return to normal
        });

        //run close preview function when the photo preview is closed by pressing the button
        //will activate camera feed once again
        closePhotoPreview.setOnClickListener(v -> {
            closePreview();
        });

        //set up a listener for the switch lens button which will change which camera is being
        //used on the phone when its clicked.
        switchLens.setOnClickListener(v -> {
            if (lensFacing == CameraSelector.DEFAULT_FRONT_CAMERA) lensFacing = CameraSelector.DEFAULT_BACK_CAMERA;
            else if (lensFacing == CameraSelector.DEFAULT_BACK_CAMERA) lensFacing = CameraSelector.DEFAULT_FRONT_CAMERA;
            // spin icon
            switchLens.animate().rotation(180-switchLens.getRotation()).start();
            showImagePreview();
        });

        //set up a listener for scaling gestures such as pinches -> this will allow for zooming capabilities
        ScaleListener listener = new ScaleListener();
        ScaleGestureDetector scaleGestureDetector = new ScaleGestureDetector(previewView.getContext(), listener);

        //set up a listener for touching the camera feed preview screen.
        previewView.setOnTouchListener((v, event) -> {
            //if the app is in drawing mode, nothing should happen for this activity when screen is tapped -> return false
            //to indicate this activity isn't handling it
            if (drawingMode)
            {
//                Log.d("TOUCH", "Passing to next listener");
                return false;
            }
//            Log.d("TOUCH", "Screen touched");
            //let scale gesture detector also know about the event in case of pinching
            scaleGestureDetector.onTouchEvent(event);

            //otherwise if the touch action is them tapping onto the screen do nothing
            if(event.getAction() == MotionEvent.ACTION_DOWN)
            {
                Log.d("TOUCH", "Pressed");
                return true;
            }
            //if the action is them removing their finger from the screen then we will autofocus to that location
            if(event.getAction() == MotionEvent.ACTION_UP)
            {
                Log.d("TOUCH", "Released");
                //final Rect sensorArraySize = cameraProviderFuture.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
                //get x and y of touch position
                float x = event.getX();
                float y = event.getY();
                String output = "x = " + x + " y = " + y;
                Log.d("TOUCH", output);

                //get display and build a meteringPoint on our screen so we have x and y in relation to our screen
                Display display = getDefaultDisplay(this);
                MeteringPointFactory factory  = new DisplayOrientedMeteringPointFactory(display, camera.getCameraInfo(), ((float) previewView.getWidth()), ((float) previewView.getHeight()));
                MeteringPoint meteringPoint = factory.createPoint(x,y);
                try
                {
                    //build a focusing area around our metering point
                    FocusMeteringAction.Builder builder = new FocusMeteringAction.Builder(meteringPoint, FocusMeteringAction.FLAG_AF);
                    builder.disableAutoCancel();
                    //focus our camera to the metering point
                    camera.getCameraControl().startFocusAndMetering(builder.build());

                    //play an animation to show what happened
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

        //Setup a touch listener for our drawing view, which simply returns the draw activities on touch event routine
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
