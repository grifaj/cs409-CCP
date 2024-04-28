from ultralytics import YOLO

"""
A script that trains from a pretrained model. 
Parameters may be adjusted to suit the need.
"""

model = YOLO('yolov8m.pt')

results = model.train(data='data.yaml', 
                      epochs=900, 
                      save_period=300,
                      imgsz=512,
                      device=0, 
                      project='./Models/',
                      verbose=True, 
                      plots=True,
                      shear=25,
                      perspective=0.001,
                      degrees=45)