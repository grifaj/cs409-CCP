package com.android.example.cpp_test;


import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import com.android.example.cpp_test.R;

public class DrawView extends View
{
    private static final String TAG = "DrawView";

    //set-up variables for tracking the drawview rectangle and dots
    private Paint paint;

    private Point[] ball_points;
    private Point start;
    private Point offset;

    private int minimumSideLength;
    private int sidew;
    private int sidel;
    private int halfCorner;
    private int cornerColor;
    private int edgeColor;
    private int outsideColor;
    private int corner = 5;

    private boolean initialized = false;
    private boolean isDrawable;

    private Drawable moveDrawable;
    private Drawable resizeDrawable1, resizeDrawable2, resizeDrawable3;

    Context context;

    //all views must setup these 4 different constructor types to intialise things about the view
    public DrawView(Context context) {
        super(context);
        this.context = context;
        init(null);
    }

    public DrawView(Context context, AttributeSet attrs) {
        super(context, attrs);
        this.context = context;
        init(attrs);
    }

    public DrawView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        this.context = context;
        init(attrs);
    }

    public DrawView(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
        super(context, attrs, defStyleAttr, defStyleRes);
        this.context = context;
        init(attrs);
    }

    //update the view to tell it whether drawmode is active or not so it knows whether to listen
    //for touch events
    public void setDrawMode(boolean drawMode)
    {
        isDrawable = drawMode;
        Log.d("DRAW", "Draw state changed to " + isDrawable);
    }

    //functions to get the bounding box measurements that can be made into a cv::rect on the c++
    //for detection purposes (gets top left co-ordinate of box, and width and height)
    public int getTopX()
    {
        return ball_points[0].x + halfCorner;
    }

    public int getTopY()
    {
        return ball_points[0].y + halfCorner;
    }

    public int getBoxWidth()
    {
        return sidew;
    }

    public int getBoxHeight()
    {
        return sidel;
    }

    //initial setup done by reading xml listing into a typedarray
    private void init(AttributeSet attrs){

        paint = new Paint();
        start = new Point();
        offset = new Point();

        TypedArray ta = context.getTheme().obtainStyledAttributes(attrs, R.styleable.DrawView, 0, 0);

        //set rectangles to minimum side lengths and set corner radius based on xml file listing
        minimumSideLength = ta.getDimensionPixelSize(R.styleable.DrawView_minimumSide, 20);
        sidew = minimumSideLength;
        sidel = minimumSideLength;
        halfCorner = (ta.getDimensionPixelSize(R.styleable.DrawView_cornersSize, 20))/2;

        //set up colours for different elements based on xml listings
        //the colour of corners, the colour of the rectangle edge
        //colour of the outside
        cornerColor = ta.getColor(R.styleable.DrawView_cornerColor, Color.BLACK);
        edgeColor = ta.getColor(R.styleable.DrawView_edgeColor, Color.WHITE);
        outsideColor = ta.getColor(R.styleable.DrawView_outsideCropColor, Color.parseColor("#00000088"));

        //initialize corners into new array of points
        ball_points = new Point[4];

        //make each of them a new point
        ball_points[0] = new Point();
        ball_points[1] = new Point();
        ball_points[2] = new Point();
        ball_points[3] = new Point();

        //get current display metrics
        DisplayMetrics metrics = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getMetrics(metrics);

        //use current display metrics to set start positions of each corner to mean the intial
        //draw rectangle is in middle of screen.
        int start_pos_x = metrics.widthPixels/2-(minimumSideLength/2);
        int start_pos_y = metrics.heightPixels/2-(minimumSideLength/2);

        //set up initial corner locations
        //top left point
        ball_points[0].x = start_pos_x;
        ball_points[0].y = start_pos_y;

        //top right point
        ball_points[1].x = start_pos_x + minimumSideLength;
        ball_points[1].y = start_pos_y;

        //bottom left point
        ball_points[2].x = start_pos_x;
        ball_points[2].y = start_pos_y + minimumSideLength;

        //bottom right point
        ball_points[3].x = start_pos_x + minimumSideLength;
        ball_points[3].y = start_pos_y + minimumSideLength;

        //get the different drawables from xml that we will draw onto screen at corner points
        moveDrawable = ta.getDrawable(R.styleable.DrawView_moveCornerDrawable);
        resizeDrawable1 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);
        resizeDrawable2 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);
        resizeDrawable3 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);

        //set corner colours
        moveDrawable.setTint(cornerColor);
        resizeDrawable1.setTint(cornerColor);
        resizeDrawable2.setTint(cornerColor);
        resizeDrawable3.setTint(cornerColor);

        //recycle attributes
        ta.recycle();

        //set initialized to true
        initialized = true;

    }

    //this is called everytime the view needs to be redrawn (on initial startup after init and
    //after invalidate() calls).
    @Override
    protected void onDraw(Canvas canvas)
    {
        super.onDraw(canvas);
        //if the view has had initial variables set up
        if(initialized)
        {
            //set paint to draw an edge
            paint.setAntiAlias(true);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeJoin(Paint.Join.ROUND);
            paint.setColor(edgeColor);
            paint.setStrokeWidth(4);

            //draw a rectangle onto the screen using corner co-ordinates and side lengths as well as the
            //paint brush we just defined
            canvas.drawRect(ball_points[0].x + halfCorner,ball_points[0].y + halfCorner, ball_points[0].x + halfCorner + sidew, ball_points[0].y + halfCorner + sidel, paint);

            //set paint to a fill style to colour in outside of the rectangle
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(outsideColor);

            //colour in above the draw rectangle
            canvas.drawRect(0, 0, canvas.getWidth(), ball_points[0].y + halfCorner, paint);
            //colour to the left of the draw rectangle
            canvas.drawRect(0, ball_points[0].y + halfCorner, ball_points[0].x + halfCorner, canvas.getHeight(), paint);
            //colour to the right of the draw rectangle
            canvas.drawRect(ball_points[0].x + halfCorner + sidew, ball_points[0].y + halfCorner, canvas.getWidth(), ball_points[0].y + halfCorner + sidel, paint);
            //colour below the draw rectangle
            canvas.drawRect(ball_points[0].x + halfCorner, ball_points[0].y + halfCorner + sidel, canvas.getWidth(), canvas.getHeight(), paint);

            //set location where each corner drawable will be between (i.e. it's position and the length of the corner).
            moveDrawable.setBounds(ball_points[0].x, ball_points[0].y, ball_points[0].x + halfCorner*2, ball_points[0].y + halfCorner*2);
            resizeDrawable1.setBounds(ball_points[1].x, ball_points[1].y, ball_points[1].x + halfCorner*2, ball_points[1].y + halfCorner*2);
            resizeDrawable2.setBounds(ball_points[2].x, ball_points[2].y, ball_points[2].x + halfCorner*2, ball_points[2].y + halfCorner*2);
            resizeDrawable3.setBounds(ball_points[3].x, ball_points[3].y, ball_points[3].x + halfCorner*2, ball_points[3].y+ halfCorner*2);

            //draw the corners onto the screen
            moveDrawable.draw(canvas);
            resizeDrawable1.draw(canvas);
            resizeDrawable2.draw(canvas);
            resizeDrawable3.draw(canvas);

        }
    }

    //used to figure out if a corner has been pressed and if so which one
    private int getCorner(float x, float y)
    {
        //for each corner check that the press was between the start of the corner position and the
        //edge of the corner
        int corner = 5;
        for (int i = 0; i < ball_points.length; i++)
        {
            float dx = x - ball_points[i].x;
            float dy = y - ball_points[i].y;
            int max = halfCorner * 2;
            if(dx <= max && dx >= 0 && dy <= max && dy >= 0){
                return i;
            }
        }
        return corner;
    }


    //gets offset between point pressed on the screen and the corner co-ordinate that is being pressed
    private Point getOffset(int left, int top, int corner)
    {
        Point offset = new Point();
        if(corner == 5)
        {
            offset.x = 0;
            offset.y = 0;
        }
        else
        {
            offset.x = left - ball_points[corner].x;
            offset.y = top - ball_points[corner].y;
        }
        return offset;
    }

    //have to override this for views, doesn't actually do anything different
    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec)
    {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
    }

    //this function handles on touch events when the DrawView has been touched/
    @Override
    public boolean onTouchEvent(MotionEvent event)
    {
        //if not in drawable mode then don't react to the touch event
        if (!isDrawable)
        {
            return false;
        }

        //otherwise we need to handle the action
        switch(event.getActionMasked())
        {
            //on initial press down
            case MotionEvent.ACTION_DOWN:
            {

                //get the coordinates of the touch event
                start.x = (int)event.getX();
                start.y = (int)event.getY();

                //find which corner has been touched
                corner = getCorner(start.x, start.y);

                //find the difference between where they touched and the actual corner point
                offset = getOffset(start.x, start.y, corner);

                //set up variables so that we account for the offset in touch and actual corner
                //point when calculating movements
                start.x = start.x - offset.x;
                start.y = start.y - offset.y;

                break;
            }
            case MotionEvent.ACTION_UP:
            {

            }
            case MotionEvent.ACTION_MOVE:
            {
                //if the event is dragging along the screen then.
                //if it is the top left corner we will attempt to move the whole rectangle.
                if(corner == 0)
                {
                    //long equations but essentially it is to ensure rules are maintained
                    //each corner co-ordinate is updated to either be where it currently is + where they have attempted to move it to
                    //or if any corners would go off the edge of the screen all balls will not be updated and positions set so that
                    //corners will stay at the edge of the screen, and side lengths will be maintained
                    ball_points[0].x = Math.max(ball_points[0].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[0].x - 2*halfCorner - sidew)), 0);
                    ball_points[1].x = Math.max(ball_points[1].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[1].x - 2*halfCorner)), sidew);
                    ball_points[2].x = Math.max(ball_points[2].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[2].x - 2*halfCorner - sidew)), 0);
                    ball_points[3].x = Math.max(ball_points[3].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[3].x - 2*halfCorner)), sidew);

                    //same is done for y co-ordinates
                    ball_points[0].y = Math.max(ball_points[0].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[0].y - 2*halfCorner - sidel)), 0);
                    ball_points[1].y = Math.max(ball_points[1].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[1].y - 2*halfCorner - sidel)), 0);
                    ball_points[2].y = Math.max(ball_points[2].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[2].y - 2*halfCorner)), sidel);
                    ball_points[3].y = Math.max(ball_points[3].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[3].y - 2*halfCorner)), sidel);

                    //set the start co-ordinates back to the new positions of the top left corner
                    start.x = ball_points[0].x;
                    start.y = ball_points[0].y;
                    //invalidate old drawing so all elements must be redrawn.
                    invalidate();
                }
                else if (corner == 1)
                {
                    //if corner is top right, allow for scaling along the width of the rectangle when movement occurs
                    //calculate new side length based on how far they move it plus the current position or update to the edge of screen if they
                    //attempt to drag off screen.
                    sidew = Math.min((Math.max(minimumSideLength, (int)(sidew + Math.floor(event.getX()) - start.x - offset.x))), sidew + (getWidth() - ball_points[1].x - 2* halfCorner));
                    //update top right and bottom right x co-ordinates
                    ball_points[1].x = ball_points[0].x + sidew;
                    ball_points[3].x = ball_points[0].x + sidew;
                    start.x = ball_points[1].x;
                    //invalidate so we redraw
                    invalidate();
                }
                else if (corner == 2)
                {
                    //if corner is bottom left, allow for scaling along the height of the rectangle when movement occurs
                    //calculate new side length based on how far they move it plus the current position or update to the edge of screen if they
                    //attempt to drag off screen.
                    sidel =  Math.min((Math.max(minimumSideLength, (int)(sidel + Math.floor(event.getY()) - start.y - offset.y))), sidel + (getHeight() - ball_points[2].y - 2* halfCorner));
                    //update bottom left and bottom right x co-ordinates
                    ball_points[2].y = ball_points[0].y + sidel;
                    ball_points[3].y = ball_points[0].y + sidel;
                    start.y = ball_points[2].y;
                    invalidate();
                }
                else if (corner == 3)
                {
                    //if corner is bottom right then allowing for scaling both on width and height
                    //do the same calculations as above to get how much to update side length by whilst
                    //still being bounded by screens edge
                    sidew = Math.min(
                                    (Math.max(minimumSideLength, (int)(sidew + Math.floor(event.getX()) - start.x - offset.x))),
                                    sidew + (getWidth() - ball_points[3].x - 2* halfCorner));
                    sidel = Math.min(
                            (Math.max(minimumSideLength, (int)(sidel + Math.floor(event.getY()) - start.y - offset.y))),
                            sidel + (getHeight() - ball_points[3].y - 2* halfCorner));

                    //update the necessary corners x and y to reflect new side lengths
                    ball_points[1].x = ball_points[0].x + sidew;
                    ball_points[3].x = ball_points[0].x + sidew;
                    ball_points[3].y = ball_points[0].y + sidel;
                    ball_points[2].y = ball_points[0].y + sidel;
                    start.x = ball_points[3].x;

                    ball_points[2].y = ball_points[0].y + sidel;
                    ball_points[3].y = ball_points[0].y + sidel;
                    ball_points[3].x = ball_points[0].x + sidew;
                    ball_points[1].x = ball_points[0].x + sidew;
                    start.y = ball_points[3].y;
                    //invalidate to redraw
                    invalidate();
                }
                break;
            }
        }
        return true;
    }
}
