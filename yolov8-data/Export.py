from ultralytics import YOLO

"""
Script that exports the Pytorch file into an ncnn file.
Ultralytics has an inbuilt function to handle this.
"""

model = YOLO('./FinalYOLOModel/weights/best.pt')

model.export(format='ncnn')
