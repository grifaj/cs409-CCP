from config import Config_2 as C
import train as train

import torch 
import torch.nn as nn
import torch.onnx as onnx 
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import random

def test_model():
    model_type = C.MODEL_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Initialise model
    model, criterion, optimiser = train.init_model(1000, model_type, pretrained=C.PRETRAINED, log=False)

    # # Load saved model parameters
    model, _ = train.load_model(
        model, 
        optimiser, 
        "/dcs/large/seal-script-project-checkpoints/mobilenet_v3_large/2024-04-17/CK-99.pt"
        # "/dcs/large/seal-script-project-checkpoints/mobilenet_v3_large/2024-04-17/CK-19.pt"
        )
        
    model.to(device)

    if model_type == 'resnet_50':
        ## Retrieve Resnet50 model weights
        weights = models.ResNet50_Weights.DEFAULT

    elif model_type == 'mobilenet_v3_large':
        ## Retrieve Mobilenet V3 Large model weights
        weights = models.MobileNet_V3_Large_Weights.DEFAULT

    elif model_type == 'vgg_19':
        weights = models.VGG19_Weights.DEFAULT

    transformation = weights.transforms()

    # Get PyTorch predictions
    csv = C.DATA_PATH
    numRands = 10
    dataset = pd.read_csv(csv, names=['image', 'label'], index_col=False)
    samples = dataset.sample(n=numRands, ignore_index=True) # Get sample of dataset with indexing reset to 0,1,...,n-1
    print(samples)
    # print(samples)

    correct = 0
    pytorch_predictions = np.zeros((numRands,))
    # tflite_predictions = np.zeros((numRands,))

    model.eval()
    for i, row in samples.iterrows():
        
        image = row['image']
        label = row['label']

        # Convert image to gray and pad to dim=3
        input = Image.open(image).convert("L")
        input = Image.fromarray(np.repeat(np.asarray(input)[..., np.newaxis], 3, -1)) # Stack grayscle images to create 3 channel image
        label = label - 1 # Cuda requires class indexing to start from 0
        value = label

        input = transformation(input)
        input = torch.unsqueeze(input, 0).to(device)
        # print(input.shape)
        label = torch.as_tensor(label)

        label = label.to(device)

        output = model(input)
        print(output)

        print(f"Nans: {torch.isnan(output).any()}")

        # Get softmax output from model
        # soft = nn.Softmax(dim=1)
        # output = soft(output)
        
        # if i == 0:
        print(f'[INFO] Testing image: {image}')
            # print(output)

        conf, preds = torch.max(output, 1)
        low_conf, _ = torch.min(output, 1)
        # prediction = int(torch.max(output, 1).cpu().numpy())
        # if i == 0:
        print(f"Prediction: {preds}")
        print(f"Actual: {value}")
        print(f"Confidence: {conf}")
        print(f"Lowest conf: {low_conf}")
        print()
        correct += torch.sum(preds == label.data)
        # prediction = random.randint(0, 1075)


    print(correct)
    print(f'[INFO] Accuracy = {round(correct.item() / numRands * 100, 2)}')

def generate_test_batch(n):
    csv = C.DATA_PATH
    numRands = 10
    dataset = pd.read_csv(csv, names=['image', 'label'], index_col=False)
    samples = dataset.sample(n=numRands, ignore_index=True) # Get sample of dataset with indexing reset to 0,1,...,n-1

if __name__=="__main__":
    test_model()