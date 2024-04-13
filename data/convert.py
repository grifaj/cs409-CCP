from config import Config_2 as C
import train

import torch 
import torch.onnx as onnx 
from torchvision import transforms, models
import numpy as np
# import onnx_tf
import pandas as pd

def convert_to_onnx(build_path, model_name):
    '''
    Convert Pytorch model to ONNX format.

    Args:
        build_path - path to store ONNX model
        model_name - name of ONNX model
    '''
    model_type = C.MODEL_NAME

    # Initialise model
    model, criterion, optimiser = train.init_model(1075, model_type, pretrained=False, log=False)

    print("[INFO] Loading model")
    # Load saved model parameters
    model, _ = train.load_model(model, optimiser, C.CONVERT_CHECKPOINT_PATH)

    # Generate dummy input to model
    x = torch.rand(1, 3, 224, 224).to("cuda")

    print("[INFO] Exporting to ONNX format")
    # Export model to onnx format
    torch_out = onnx.export(model, x, f"{build_path}/{model_name}.onnx", export_params=True)

if __name__ == "__main__":
    # Convert Pytorch model to onnx format
    convert_to_onnx(build_path, model_name)