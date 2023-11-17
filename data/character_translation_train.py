# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from character_translation_model import LeNet
from character_translation_load import DatasetLoad
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import os
from PIL import Image
import pandas as pd
import math

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
ap.add_argument("-i", "--index", type=int, required=True,
	help="train model on characters with index 1..i")
args = vars(ap.parse_args())
print(args["index"])

data_dir = './source'
csvFile = 'trainData.csv'

# Data must be in CustomDataset format for PyTorch Dataloader wrapper object
class CustomDataset(Dataset):
    def __init__(self, csvFile=csvFile, transform=None, target_transform=None):
        self.data = pd.read_csv(os.path.join(data_dir, csvFile), sep=",", names = ["img", "label"])
        self.transform = transform
        self.target_transform = target_transform
        # self.labels = y
        # self.x = X

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data["img"][idx]).convert("L")
        label = self.data["label"][idx]

        if self.transform is not None:
            img = self.transform(img)

        
        return (img, label)

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the dataset
print("[INFO] loading the dataset...")

datasetLoader = DatasetLoad(data_dir, args["index"], csvFile)
datasetLoader.createCsv()

transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

dataset = CustomDataset(csvFile,transform=transform)
# print(dataset.__getitem__(0))
nontest_set, test_set = random_split(dataset,[math.ceil(dataset.__len__()*TRAIN_SPLIT), math.floor(dataset.__len__()*VAL_SPLIT)])
# print("== TEST SET ==")
# print(test_set[0][0][0].shape)
# print(test_set[0][0].shape)
# print("== NONTEST SET ==")
# print(nontest_set)

train_set, validation_set = random_split(nontest_set,[math.ceil(nontest_set.__len__()*TRAIN_SPLIT), math.floor(nontest_set.__len__()*VAL_SPLIT)])
trainDataLoader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE)
validationDataLoader = DataLoader(dataset=validation_set, shuffle=True, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(dataset=test_set, shuffle=True, batch_size=BATCH_SIZE)


# Split the dataset into testing and non-testing subsets
# X_nontest, X_test, y_nontest, y_test = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42, shuffle=True)

# Split the non-testing data into training and validation subsets
# X_train, X_val, y_train, y_val = train_test_split(X_nontest, y_nontest, test_size=VAL_SPLIT, random_state=42, shuffle=True)

# Wrap the data in CustomDataset class
# trainData = CustomDataset(X_train, y_train)
# valData = CustomDataset(X_val, y_val)
# testData = CustomDataset(X_test, y_test)


# initialize the train, validation, and test data loaders
# trainDataLoader = DataLoader(trainData, shuffle=True,
# 	batch_size=BATCH_SIZE)
# valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
# testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(validationDataLoader.dataset) // BATCH_SIZE


# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(
	numChannels=1,
	classes=args["index"]).to(device)
# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()
# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0
    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # print(x.shape)
        # print(x)
        # perform a forward pass and calculate the training loss
        pred = model(x)
        # subtract 1 from all ground trutch class values as pred indexes from 0 
        y_array = torch.Tensor.numpy(y)
        y_array = y_array - 1
        y = torch.from_numpy(y_array)
        
        loss = lossFn(pred, y)
        print(f"== LOSS: {loss}")
        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
		

# switch off autograd for evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    # loop over the validation set
    for (x, y) in validationDataLoader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # make the predictions and calculate the validation loss
        pred = model(x)
        totalValLoss += lossFn(pred, y)
        # calculate the number of correct predictions
        valCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()
        
# calculate the average training and validation loss
avgTrainLoss = totalTrainLoss / trainSteps
avgValLoss = totalValLoss / valSteps
# calculate the training and validation accuracy
trainCorrect = trainCorrect / len(trainDataLoader.dataset)
valCorrect = valCorrect / len(validationDataLoader.dataset)
# update our training history
H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
H["train_acc"].append(trainCorrect)
H["val_loss"].append(avgValLoss.cpu().detach().numpy())
H["val_acc"].append(valCorrect)
# print the model training and validation information
print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
    avgTrainLoss, trainCorrect))
print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
    avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []
	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)
		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())
# generate a classification report
print(classification_report(test_set.targets.cpu().numpy(),
	np.array(preds), target_names=test_set.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
# serialize the model to disk
torch.save(model, args["model"])