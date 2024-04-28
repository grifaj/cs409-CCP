<div align="center">

<img src="https://github.com/grifaj/cs409-CCP/assets/17861497/96fec4b0-3a4f-4e5e-9dea-bccb3126933c">

</div>


# cs409-CCP | Object Detection

This directory lists all of the source files that was used during the creation of the YOLOv8 model for detecting seal script characters. There are three key sections in this directory:

1. Data Generation Notebook
2. YOLOv8 Data
3. Hyper-parameter Testing

## Data Generation Notebook

This notebook contains all the techniques and and functions used to generate the data needed to train the YOLOv8 model. It requires certain directories like the raw seal script character set to be ran correctly, which is not included here because it will already be uploaded in a separate directory. Detailed explanations on how everything works can be found in the notebook.

The directory `./background/`, contains all the background images that were used during the data generation process.

## YOLOv8 Data

This directory contains all the necessary information needed to train the object detection model. It contains:

1. The training, validation and testing data sets along with their annotations
2. The final model
    - Plots generated during training
    - Results from training
    - Image batches
    - Model weights
3. Export and Train scripts

Note that the actual ncnn model is not included in this directory because it is too large, but it does contain the Pytorch version to be used in the testing scripts. The ncnn model can be found in the app itself.

## Hyper-parameter Testing

This directory consists of files that were used during the hyper-parameter investigation of the confidence threshold *c*. It contains:

1. Control Images
2. Test Images
3. Labels for the bounding boxes of the Test Images
4. Test script to run the investigation
5. Results stored in a csv file

The results are plotted in the Data Generation notebook and can be found in the group report.