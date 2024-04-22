from config import Config as C
import train as train
import torch 
import torch.nn as nn
import torch.onnx as onnx 
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import random

def test_model(num_samples, model_path):
    ''' Main function for testing model model_path. Contains testing loop and reports accuracy obtained on num_samples images '''
    model_type = C.MODEL_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialise model
    model, criterion, optimiser = train.init_model(C.NUM_CLASSES, model_type, pretrained=C.PRETRAINED, log=False)

    # Load saved model parameters
    model, _ = train.load_model(
        model, 
        optimiser, 
        model_path
        )
    
    model.classifier = nn.Sequential(model.classifier, nn.Softmax(dim=1))

    print(model)
        
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

    # Read dataset csv to generate samples to test with
    csv = C.DATA_PATH
    numRands = num_samples # Number of test samples
    dataset = pd.read_csv(csv, names=['image', 'label'], index_col=False)
    samples = dataset.sample(n=numRands, ignore_index=True) # Get sample of dataset with indexing reset to 0,1,...,n-1

    correct = 0
    pytorch_predictions = np.zeros((numRands,))

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

        label = torch.as_tensor(label)
        label = label.to(device)

        output = model(input)

        print(f'[INFO] Testing image: {image}')

        conf, preds = torch.max(output, 1)
        low_conf, _ = torch.min(output, 1)

        print(f"Prediction: {preds}")
        print(f"Actual: {value}")
        print(f"Confidence: {conf}")
        print(f"Lowest conf: {low_conf}")
        print()

        correct += torch.sum(preds == label.data)

    print(f'[INFO] Accuracy = {round(correct.item() / numRands * 100, 2)}')

if __name__=="__main__":
    test_model(num_samples=1000, model_path="/dcs/large/seal-script-project-checkpoints/mobilenet_v3_large/2024-04-19/CK-86.pt")