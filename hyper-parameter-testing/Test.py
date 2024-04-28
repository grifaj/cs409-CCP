import numpy as np
from ultralytics import YOLO
import glob
import re
import csv

"""
This script is used to test the trained YOLO model with different values of the hyper-parameter confidence threshold c.
The results are computed and stored in a csv file with the corresponding columns: confidence, precision, recall, control_fp.
Precision and recall are computed for the labelled test data located in './images/' and the control false positives are
computed using the control images in './control-images/'. The results are plotted in the Data Generation notebook.
"""

def iou(boxA, boxB):
    """
    
    Computes the Intersection over Union of two bounding boxes. Uses the definition of an
    intersection rectangle to find the overlapping area.

    Args:
        boxA numpy.ndarray: The array (x, y, w, h) describing the first bounding box in YOLOv8 format.
        boxB numpy.ndarray: The array (x, y, w, h) describing the second bounding box in YOLOv8 format.

    Returns:
        float: The IoU computed.
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2) which corresponds to the top left (x1, y1) and bottom right (x2, y2) corner coordinates
    boxA = [boxA[0] - boxA[2] / 2, boxA[1] - boxA[3] / 2, boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2]
    boxB = [boxB[0] - boxB[2] / 2, boxB[1] - boxB[3] / 2, boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2]
    
    # Determine the (x, y)-coordinates of the intersection rectangle
    
    # Taking the inner top left coordinate
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    
    # Taking the inner bottom right coordinate
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def compute_precision_recall(boxesA, boxesB):
    """
    
    Computes the Precision and Recall of two lists of bounding
    boxes using IoU with a threshold of 0.5.

    Args:
        boxesA list: List containing all the bounding boxes for the ground truth
        boxesB list: List containing all the predicted bounding boxes

    Returns:
        tuple: Precision and Recall value pair
    """
    
    tp = 0
    
    # Loop through the boxes in the ground truth and compare it with the boxes in the predictions
    for i in range(boxesA.shape[0]):
        for j in range(boxesB.shape[0]):
            # It declares a match if the IoU is greater or equal to 0.5
            if iou(boxesA[i,:], boxesB[j,:]) >= 0.5:
                tp += 1
                break
            
    # If there are no matches, returns a precision and recall value of 0
    if boxesB.shape[0] == 0:
        return 0,0
    
    # Otherwise uses the formula: TP / number of predicted boxes, TP / number of actual boxes
    return tp / boxesB.shape[0], tp / boxesA.shape[0]

def main():

    csv_file = './results.csv'

    headers = ['confidence', 'precision', 'recall', 'control_fp']

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # Loading the model at the best epoch
    model = YOLO('../yolov8-data/FinalYOLOModel/weights/best.pt')

    # Retrieving image paths for testing and control images
    images = glob.glob('./images/*')
    control_images = glob.glob('./control-images/*')
    
    # RegEx pattern to retrieve the number of the test image for finding its annotated data
    pattern = r"(?<=\/)[\w\d]+(?=\.jpg)"

    # Considering values of c = 0 to c = 1 in increments of 0.1
    for i in range(0, 101):
        conf = i / 100
        
        # Variables to store the cumulative precision and recall
        precision = 0
        recall = 0

        # First testing on the annotated images
        for image in images:
            
            # Results are returned in normalised (x, y, w, h) parameters where (x, y) is the centre of the bounding box
            results = model.predict(image, imgsz=512, conf=conf)
            xywhn = results[0].boxes.numpy().xywhn
            
            # Finding and processing the annotated data in YOLOv8 format into an array
            im_name = re.findall(pattern, image)[0]
            labels = []
            with open(f'./labels/{im_name}.txt', 'r') as file:
                for line in file:
                    columns = line.strip().split()
                    labels.append([float(col) for col in columns[1:]])
            labels = np.array(labels)
            
            # Computes the precision and recall and adds to the cumulative variables
            pr = compute_precision_recall(labels, xywhn)
            precision += pr[0]
            recall += pr[1]
        
        # Computes the average PR for that confidence threshold
        avg_precision = precision / len(images)
        avg_recall = recall / len(images)
        
        
        # Next, tests the model on the control images and counts the false positives
        false_positives = 0

        for image in control_images:
            results = model.predict(image, imgsz=512, conf=conf)
            xywhn = results[0].boxes.numpy().xywhn
            # Any result is a false positive
            false_positives += len(xywhn)
        
        # Writing the results for this confidence threshold to the csv file
        data = (conf, avg_precision, avg_recall, false_positives)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
            
if __name__ == "__main__":
    main()