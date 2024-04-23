<div align="center">

<img src="https://github.com/grifaj/cs409-CCP/assets/17861497/96fec4b0-3a4f-4e5e-9dea-bccb3126933c">

</div>


# cs409-CCP | Android Application

Setup instrucitons:

Install android studio: https://developer.android.com/studio, it could take a while

Select File -> New -> Project from version control

Enter the url of the repositiory: git@github.com:grifaj/cs409-CCP.git or login to Github and select the repo

Switch to the app branch with `git switch android-app` in the inbuilt termial

Select File -> Sync Project with Gradle Files to build the enviroment

Press run, you may need to select  "File -> Invalidate Caches -> Invalidate and Restart" to get it to work

Bounding boxes are created by calling enhance.cpp in the cpp folder, specificaly showImage2. The prototype of the function should be cv::Mat showImage2(cv::Mat mat) where the mat is the image from the camera and to be returned
