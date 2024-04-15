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

    public void setDrawMode(boolean drawMode)
    {
        isDrawable = drawMode;
        Log.d("DRAW", "Draw state changed to " + isDrawable);
    }

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

    private void init(AttributeSet attrs){

        paint = new Paint();
        start = new Point();
        offset = new Point();

        TypedArray ta = context.getTheme().obtainStyledAttributes(attrs, R.styleable.DrawView, 0, 0);

        //initial dimensions
        minimumSideLength = ta.getDimensionPixelSize(R.styleable.DrawView_minimumSide, 20);
        sidew = minimumSideLength;
        sidel = minimumSideLength;
        halfCorner = (ta.getDimensionPixelSize(R.styleable.DrawView_cornersSize, 20))/2;

        //colors
        cornerColor = ta.getColor(R.styleable.DrawView_cornerColor, Color.BLACK);
        edgeColor = ta.getColor(R.styleable.DrawView_edgeColor, Color.WHITE);
        outsideColor = ta.getColor(R.styleable.DrawView_outsideCropColor, Color.parseColor("#00000088"));

        //initialize corners;
        ball_points = new Point[4];

        ball_points[0] = new Point();
        ball_points[1] = new Point();
        ball_points[2] = new Point();
        ball_points[3] = new Point();

        DisplayMetrics metrics = new DisplayMetrics();
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        windowManager.getDefaultDisplay().getMetrics(metrics);

        int start_pos_x = metrics.widthPixels/2-(minimumSideLength/2);
        int start_pos_y = metrics.heightPixels/2-(minimumSideLength/2);

        //init corner locations;
        //top left
        ball_points[0].x = start_pos_x;
        ball_points[0].y = start_pos_y;

        //top right
        ball_points[1].x = start_pos_x + minimumSideLength;
        ball_points[1].y = start_pos_y;

        //bottom left
        ball_points[2].x = start_pos_x;
        ball_points[2].y = start_pos_y + minimumSideLength;

        //bottom right
        ball_points[3].x = start_pos_x + minimumSideLength;
        ball_points[3].y = start_pos_y + minimumSideLength;

        //init drawables
        moveDrawable = ta.getDrawable(R.styleable.DrawView_moveCornerDrawable);
        resizeDrawable1 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);
        resizeDrawable2 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);
        resizeDrawable3 = ta.getDrawable(R.styleable.DrawView_resizeCornerDrawable);

        //set drawable colors
        moveDrawable.setTint(cornerColor);
        resizeDrawable1.setTint(cornerColor);
        resizeDrawable2.setTint(cornerColor);
        resizeDrawable3.setTint(cornerColor);

        //recycle attributes
        ta.recycle();

        //set initialized to true
        initialized = true;

    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        super.onDraw(canvas);
        //set paint to draw edge, stroke
        if(initialized)
        {
            paint.setAntiAlias(true);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeJoin(Paint.Join.ROUND);
            paint.setColor(edgeColor);
            paint.setStrokeWidth(4);

            //crop rectangle
            canvas.drawRect(ball_points[0].x + halfCorner,ball_points[0].y + halfCorner, ball_points[0].x + halfCorner + sidew, ball_points[0].y + halfCorner + sidel, paint);

            //set paint to draw outside color, fill
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(outsideColor);

            //top rectangle
            canvas.drawRect(0, 0, canvas.getWidth(), ball_points[0].y + halfCorner, paint);
            //left rectangle
            canvas.drawRect(0, ball_points[0].y + halfCorner, ball_points[0].x + halfCorner, canvas.getHeight(), paint);
            //right rectangle
            canvas.drawRect(ball_points[0].x + halfCorner + sidew, ball_points[0].y + halfCorner, canvas.getWidth(), ball_points[0].y + halfCorner + sidel, paint);
            //bottom rectangle
            canvas.drawRect(ball_points[0].x + halfCorner, ball_points[0].y + halfCorner + sidel, canvas.getWidth(), canvas.getHeight(), paint);

            //set bounds of drawables
            moveDrawable.setBounds(ball_points[0].x, ball_points[0].y, ball_points[0].x + halfCorner*2, ball_points[0].y + halfCorner*2);
            resizeDrawable1.setBounds(ball_points[1].x, ball_points[1].y, ball_points[1].x + halfCorner*2, ball_points[1].y + halfCorner*2);
            resizeDrawable2.setBounds(ball_points[2].x, ball_points[2].y, ball_points[2].x + halfCorner*2, ball_points[2].y + halfCorner*2);
            resizeDrawable3.setBounds(ball_points[3].x, ball_points[3].y, ball_points[3].x + halfCorner*2, ball_points[3].y+ halfCorner*2);

            //place corner drawables
            moveDrawable.draw(canvas);
            resizeDrawable1.draw(canvas);
            resizeDrawable2.draw(canvas);
            resizeDrawable3.draw(canvas);

        }
    }

    private int getCorner(float x, float y)
    {
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

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec)
    {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event)
    {
        if (!isDrawable)
        {
            return false;
        }
        switch(event.getActionMasked())
        {
            case MotionEvent.ACTION_DOWN:
            {

                //get the coordinates
                start.x = (int)event.getX();
                start.y = (int)event.getY();

                //get the corner touched if any
                corner = getCorner(start.x, start.y);

                //get the offset of touch(x,y) from corner top-left point
                offset = getOffset(start.x, start.y, corner);

                //account for touch offset in starting point
                start.x = start.x - offset.x;
                start.y = start.y - offset.y;

                break;
            }
            case MotionEvent.ACTION_UP:
            {

            }
            case MotionEvent.ACTION_MOVE:
            {
                if(corner == 0)
                {
                    ball_points[0].x = Math.max(ball_points[0].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[0].x - 2*halfCorner - sidew)), 0);
                    ball_points[1].x = Math.max(ball_points[1].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[1].x - 2*halfCorner)), sidew);
                    ball_points[2].x = Math.max(ball_points[2].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[2].x - 2*halfCorner - sidew)), 0);
                    ball_points[3].x = Math.max(ball_points[3].x + (int) Math.min(Math.floor((event.getX() - start.x - offset.x)), Math.floor(getWidth() - ball_points[3].x - 2*halfCorner)), sidew);

                    ball_points[0].y = Math.max(ball_points[0].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[0].y - 2*halfCorner - sidel)), 0);
                    ball_points[1].y = Math.max(ball_points[1].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[1].y - 2*halfCorner - sidel)), 0);
                    ball_points[2].y = Math.max(ball_points[2].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[2].y - 2*halfCorner)), sidel);
                    ball_points[3].y = Math.max(ball_points[3].y + (int) Math.min(Math.floor((event.getY() - start.y - offset.y)), Math.floor(getHeight() - ball_points[3].y - 2*halfCorner)), sidel);

                    start.x = ball_points[0].x;
                    start.y = ball_points[0].y;
                    invalidate();
                }
                else if (corner == 1)
                {
                    sidew = Math.min((Math.max(minimumSideLength, (int)(sidew + Math.floor(event.getX()) - start.x - offset.x))), sidew + (getWidth() - ball_points[1].x - 2* halfCorner));
                    ball_points[1].x = ball_points[0].x + sidew;
                    ball_points[3].x = ball_points[0].x + sidew;
                    start.x = ball_points[1].x;
                    invalidate();
                }
                else if (corner == 2)
                {
                    sidel =  Math.min((Math.max(minimumSideLength, (int)(sidel + Math.floor(event.getY()) - start.y - offset.y))), sidel + (getHeight() - ball_points[2].y - 2* halfCorner));
                    ball_points[2].y = ball_points[0].y + sidel;
                    ball_points[3].y = ball_points[0].y + sidel;
                    start.y = ball_points[2].y;
                    invalidate();
                }
                else if (corner == 3)
                {
                    sidew = Math.min(
                                    (Math.max(minimumSideLength, (int)(sidew + Math.floor(event.getX()) - start.x - offset.x))),
                                    sidew + (getWidth() - ball_points[3].x - 2* halfCorner));
                    sidel = Math.min(
                            (Math.max(minimumSideLength, (int)(sidel + Math.floor(event.getY()) - start.y - offset.y))),
                            sidel + (getHeight() - ball_points[3].y - 2* halfCorner));
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
                    invalidate();
                }
                break;
            }
        }
        return true;
    }
}
