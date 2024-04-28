import numpy as np
from PIL import Image
import os
import glob
import csv
import pandas as pd
from tqdm import tqdm
from config import Config as C
import logging
from datetime import datetime
import sys
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# initialise logger
logging.basicConfig(filename=C.LOG_PATH, encoding="utf-8", level=logging.INFO, 
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("==========Logger started===========")

## Initialize device for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(torch.cuda.is_available())

# change model download path to /large
if C.TORCH_MODEL_CACHE:
    os.environ["TORCH_HOME"] = C.TORCH_MODEL_CACHE

## Initialize data
class CharactersDataSet(Dataset):
  """
  This is a custom dataset class for loading images and labels.
  """
  def __init__(self, x, y, transform=None):
    self.x = x
    self.y = y
    self.transform = transform

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, index):
    
    image = Image.open(self.x.iloc[index]).convert("L")
    image = Image.fromarray(np.repeat(np.asarray(image)[..., np.newaxis], 3, -1)) # Stack grayscle images to create 3 channel image
    label = self.y.iloc[index]-1 # Cuda requires class indexing to start from 0
    
    if self.transform:
        image = self.transform(image)

    return image, label


def init_dataset(model_type):
    '''
        Initializes dataset and dataloaders used for training model. Reads data from CSV file specified in config. Splits data into train and validation sets.
    '''

    logging.info("Loading dataset")

    if not os.path.exists(C.DATA_PATH):
        print('[INFO] Data csv does not exist.')
        return None, None
    else:
        data = pd.read_csv(C.DATA_PATH, names=['file_path', 'label'])
        x = data['file_path']
        y = data['label']
        num_classes = np.unique(y)[-1]
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=C.TEST_SIZE, stratify=y)

        if model_type == 'resnet_50':
            weights = models.ResNet50_Weights.DEFAULT           
        elif model_type == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        elif model_type == 'vgg_19':
            weights = models.VGG19_BN_Weights.DEFAULT

        transformation = weights.transforms()

        train_data_object = CharactersDataSet(x_train, y_train, transformation)
        val_data_object = CharactersDataSet(x_val, y_val, transformation)

        train_loader = torch.utils.data.DataLoader(train_data_object,
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        val_loader = torch.utils.data.DataLoader(val_data_object,
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        
        dataloaders = {'train': train_loader, 'validation': val_loader}
        datasets = {'train': train_data_object, 'validation': val_data_object}

    return dataloaders, datasets, num_classes


def init_model(num_classes, model_type, pretrained=True, log=True): #, use_cpu=False, pretrained=False):
    '''
    Initialize device for cuda and load specified model
    '''
    
    if model_type == 'resnet_50':
        ## Retrieve Resnet50 model
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights = weights)
        else:
            model = models.resnet50()
    elif model_type == 'mobilenet_v3_large':
        ## Retrieve Mobilenet V3 Large model
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            model = models.mobilenet_v3_large(weights = weights)
        else:
            model = models.mobilenet_v3_large()
    elif model_type == 'vgg_19':
        if pretrained:
            weights = models.VGG19_BN_Weights.DEFAULT
            model = models.vgg19_bn(weights = weights)
        else:
            model = models.vgg19_bn()

    if pretrained:
        for param in model.parameters():
            param.requires_grad = False 

    # Create classifcation layer for chosen model type
    if model_type == 'resnet_50':
        model.fc = nn.Sequential(
                    nn.Linear(2048, 1600),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(1600, 512),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),    
                    nn.Dropout(p=0.2),
                    nn.Linear(512, num_classes)
        ).to(device)
    elif model_type == 'mobilenet_v3_large':
        model.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        ).to(device)
    elif model_type == 'vgg_19':
        model.classifier[6] = nn.Linear(4096, num_classes, True).to(device)
        model.classifier = nn.Sequential(
            model.classifier
        )
    
    criterion = nn.CrossEntropyLoss()
    
    optimisers = [
        optim.Adam(model.parameters(), lr=C.LEARNING_RATE, betas=C.ADAM_BETA),
    ]

    model.to(device)

    logging.info(f"Using device: {device.type}")
    logging.info(f"Pretrained: {pretrained}")

    if log:
        logging.info(model)

    return model, criterion, optimisers


def save_model(model:nn.Module, optimisers, epoch:int):
    ''' Save the model state, optimizer state and epoch so that they can be loaded in later to resume training '''
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_a": optimisers[0].state_dict()
    }, C.CHECKPOINT_PATH + f"CK-{epoch}.pt")


def load_model(model:nn.Module, optimisers, load_from:str):
    ''' Load the model from load_from path '''
    logging.info(f"Loading checkpoint {load_from}")
    checkpoint = torch.load(load_from)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimisers[0].load_state_dict(checkpoint["optimiser_a"])
    epoch = checkpoint["epoch"]
    logging.info("Loading done")
    
    return model, epoch

## Train model
def train_model(model, dataloaders, datasets, optimisers, criterion, start_epoch):
    ''' Main training function containing training and valiation loop '''
    def zero_grads(optimisers):
        for o in optimisers: o.zero_grad()
        
    def steps(optimisers):
        for o in optimisers: o.step()

    if C.CHECKPOINT_PATH[-1] != "/": C.CHECKPOINT_PATH += "/"
    C.CHECKPOINT_PATH += datetime.today().strftime('%Y-%m-%d') + "/"
    os.makedirs(C.CHECKPOINT_PATH, exist_ok=True)
    logging.debug("Checkpoint files ok")
    
    logging.info("Starting training")
    for epoch in range(start_epoch, C.EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, C.EPOCHS))

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Training phase")
            else:
                model.eval()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Validation phase")

            running_loss = 0.0
            running_corrects = 0

            # For all batches in phase
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # Update model if in training phase
                if phase == 'train':
                    zero_grads(optimisers)
                    loss.backward()
                    steps(optimisers)

                # Get predicted class and calculate performance metrics for batch
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate performance metrics for epoch
            epochLoss = running_loss / datasets[phase].__len__()
            epochAccuracy = running_corrects.double() / datasets[phase].__len__()

            logging.info(f"[Epoch {epoch+1}/{C.EPOCHS}] {phase} [Loss {epochLoss:.4f}] [Accuracy {epochAccuracy:.4f}]")

            if phase == "train" and (epoch + 1) % C.SAVE_EVERY_N == 0:
                    save_model(model, optimisers, epoch)
    
    return model, optimisers, epoch


def main(): 
    pretrained = C.PRETRAINED
    # Select the model type to be trained
    model_type = C.MODEL_NAME
    logging.info("Loading dataset")
    logging.info("Loading data loaders")
    dataloaders, datasets, num_classes = init_dataset(model_type)
    logging.info(f"Num classes {num_classes}")
    if not num_classes == C.NUM_CLASSES:
        logging.critical(f"Number of detected classes does not agree with config file")
        sys.exit()
    logging.info("Done")
    epoch=0
    logging.info(f"Loading {model_type}")
    model, criterion, optimisers = init_model(num_classes, model_type, pretrained=pretrained, log=True)
    if C.LOAD_CHECKPOINT_PATH != "":
        model, epoch = load_model(model, optimisers, C.LOAD_CHECKPOINT_PATH)  
    logging.info("Done")
    
    logging.info("Start training")
    model, optimisers, epoch = train_model(model, dataloaders, datasets, optimisers, criterion, epoch)
    logging.info("Finished, saving...")
    save_model(model, optimisers, epoch)
    logging.info("Exiting")

if __name__=="__main__":
    main()