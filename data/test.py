from config import Config_resnet as C
import resnet_50 as resnet50

import torch 
import torch.onnx as onnx 
from torchvision import datasets, models, transforms
import numpy as np
import pandas as pd
from PIL import Image
import random
from onnx_tf.backend import prepare

def convert_to_tflite():
    # Load saved model parameters
    model, _ = resnet50.load_model(model, optimiser, "/dcs/large/seal-script-project-checkpoints/resnet50/2024-02-28/CK-139.pt")

    # Generate dummy input to model
    x = torch.rand(1, 3, 224, 224).to("cuda")

    # Export model to onnx format
    torch_out = onnx.export(model, x, "build/seals-resnet50.onnx", export_params=True)

    # Load onnx file and convert to tensorflow lite format
    onnx_model = onnx.load("build/seals-resnet50.onnx")

    tf_model = prepare(onnx_model)

    tf_model.export_graph("build/resnet50.tf")

    converter = tf.lite.TFLiteConverter.from_saved_model("build/resnet50.tf")
    tflite_model = converter.convert()
    open('build/resnet18.tflite', 'wb').write(tflite_model)

def test_tflite():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Initialise model
    model, criterion, optimiser = resnet50.init_model(1076, pretrained=False, log=False)

    # # Load saved model parameters
    model, _ = resnet50.load_model(
        model, 
        optimiser, 
        "/dcs/large/seal-script-project-checkpoints/resnet50/2024-02-28/CK-139.pt"
        ).to(device)

    weights = models.ResNet50_Weights.DEFAULT
    transformation = weights.transforms()

    # Get PyTorch predictions
    csv = 'trainData.csv'
    numRands = 1000
    dataset = pd.read_csv(csv, names=['image', 'label'], index_col=False)
    samples = dataset.sample(n=numRands, ignore_index=True) # Get sample of dataset with indexing reset to 0,1,...,n-1

    print(samples)

    correct = 0
    pytorch_predictions = np.zeros((numRands,))
    tflite_predictions = np.zeros((numRands,))
    for i, row in samples.iterrows():

        image = row['image']
        label = row['label']
        print(f'[INFO] Testing image: {image}, label: {label}')

        # Convert image to gray and pad to dim=3
        input = Image.open(image).convert("L")
        input = Image.fromarray(np.repeat(np.asarray(input)[..., np.newaxis], 3, -1)) # Stack grayscle images to create 3 channel image
        label = label - 1 # Cuda requires class indexing to start from 0

        input = transformation(input).to(device)
        label = label.to(device)

        output = model(input)

        prediction = int(torch.max(output.data, 1)[1].numpy())
        # prediction = random.randint(0, 1075)

        pytorch_predictions[i] = prediction


    print(f'[INFO] Accuracy = {round(correct / numRands * 100, 2)}')


if __name__=="__main__":
    convert_to_tflite()
    test_tflite()