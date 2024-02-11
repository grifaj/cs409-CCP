import numpy as np
from PIL import Image
import os
import glob
import csv
import pandas as pd
# import traceback
# import argparse
# from tqdm import tqdm
from config import Config_resnet as C
import logging
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
# from torchsummary import summary


# initialise logger
logging.basicConfig(filename=C.LOG_PATH, encoding="utf-8", level=logging.INFO, 
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("==========Logger started===========")

# change model download path to /large
if C.TORCH_MODEL_CACHE:
    os.environ["TORCH_HOME"] = C.TORCH_MODEL_CACHE

## Initialize data
class CharactersDataSet(Dataset):
  """
  This is a custom dataset class.
  """
  def __init__(self, x, y, transform=None):
    self.x = x
    self.y = y
    self.transform = transform

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    image = Image.open(self.x.iloc[index]).convert("L")
    label = self.y.iloc[index]-1 # Cuda requires class indexing to start from 0
    
    if self.transform:
        image = self.transform(image)

    return image, label
  

def init_dataset():
    '''
        Initializes dataset and dataloaders used for training model. Reads data from CSV file <data_file>. Splits data into train, validation and test sets.
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
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=C.TEST_SIZE, random_state=4)
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=C.TEST_SIZE, random_state=4)


        transformation = transforms.Compose([
                transforms.Resize((C.IMAGE_SIZE,C.IMAGE_SIZE)),
                transforms.ToTensor(),
        #         transforms.Normalize(mean=MEAN, std=STD),
            ])

        train_data_object = CharactersDataSet(x_train, y_train, transformation)
        val_data_object = CharactersDataSet(x_val, y_val, transformation)
        # test_data_object = CharactersDataSet(x_test, y_test, transformation)

        train_loader = torch.utils.data.DataLoader(train_data_object,
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        val_loader = torch.utils.data.DataLoader(val_data_object,
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        # test_loader = torch.utils.data.DataLoader(test_data_object,
        #                                         batch_size=C.BATCH_SIZE,
        #                                         shuffle=C.SHUFFLE_DATA)
        
        dataloaders = {'train': train_loader, 'validation': val_loader} #, 'test': test_loader}
        datasets = {'train': train_data_object, 'validation': val_data_object} # 'test': test_data_object}

    return dataloaders, datasets, num_classes


def init_model(num_classes): #, use_cpu=False, pretrained=False):
    '''
    Initialize device for cuda and load resnet50 model.
    '''
    ## Initialize device for cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # if use_cpu:
        # device = torch.device("cpu")
    
    ## Retrieve Resnet50 model
    # if pretrained:
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device)
    # else:
    #     model = models.resnet50().to(device)

    
        
    for param in model.parameters():
        param.requires_grad = False  
        
    if device.type == "cuda":
        model.cuda()

    logging.info(f"Using device: {device.type}")

    ## Initialize model
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

    # numFcInputs = model.fc[0].in_features

    model.fc = nn.Sequential(
                nn.Linear(2048, 1600),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1600, num_classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimisers = [
        optim.Adam(model.parameters(), lr=C.LEARNING_RATE, betas=C.ADAM_BETA),
    ]

    return model, criterion, optimisers, device


## Save model dictionaries
def save_model(model:nn.Module, optimisers, epoch:int):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_a": optimisers[0].state_dict()
    }, C.CHECKPOINT_PATH + f"CK-{epoch}.pt")

def load_model(model:nn.Module, optimisers, load_from:str):
    logging.info(f"Loading checkpoint {load_from}")
    checkpoint = torch.load(load_from)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimisers[0].load_state_dict(checkpoint["optimiser_a"])
    epoch = checkpoint["epoch"]
    logging.info("Loading done")
    
    return epoch


## Train model
def train_model(model, dataloaders, datasets, optimisers, criterion, epoch, device):
    def zero_grads(optimisers):
        for o in optimisers: o.zero_grad()
        
    def steps(optimisers):
        for o in optimisers: o.step()
    # if debug:
    #     file_name = 'debug.txt'
    # else:
        # file_name = f'{C.MODEL_NAME}_B{C.BATCH_SIZE}_E{C.EPOCHS}_I{C.IMAGE_SIZE}_N{num_examples}.txt' # make file to store training results
    
    if C.CHECKPOINT_PATH[-1] != "/": C.CHECKPOINT_PATH += "/"
    C.CHECKPOINT_PATH += datetime.today().strftime('%Y-%m-%d') + "/"
    os.makedirs(C.CHECKPOINT_PATH, exist_ok=True)
    logging.debug("Checkpoint files ok")
    
    # print(f'[INFO] New file created: {file_name}')
    # with open(os.path.join(training_results_dir, file_name), 'w') as f:
    logging.info("Starting training")
    for epoch in range(C.EPOCHS):
        print('Epoch {}/{}'.format(epoch+1, C.EPOCHS))
        # f.write('Epoch {}/{}\n'.format(epoch+1, C.EPOCHS))
        # print('-' * 10)
        # f.write('-' * 10+'\n')

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Training phase")
            else:
                model.eval()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Validation phase")

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    zero_grads(optimisers)
                    loss.backward()
                    steps(optimisers)


                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epochLoss = running_loss / datasets[phase].__len__()
            epochAccuracy = running_corrects.double() / datasets[phase].__len__()

            logging.info(f"[Epoch {epoch+1}/{C.EPOCHS}] {phase} [Loss {epochLoss:.4f}] [Accuracy {epochAccuracy:.4f}]")

            if phase == "train" and (epoch + 1) % 20 == 0:
                    save_model(model, optimisers, epoch)
    
    return model


def main(): 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-pretrained", action="store_true")
    # parser.add_argument("-use_cpu", action="store_true")

    # args = parser.parse_args()

    # pretrained = args.pretrained
    # use_cpu = args.use_cpu

    logging.info("Loading dataset")
    logging.info("Loading data loaders")
    dataloaders, datasets, num_classes = init_dataset()
    logging.info(f"Num classes {num_classes}")
    logging.info("Done")
    epoch=0
    logging.info("Loading ResNet50")
    model, criterion, optimisers, device = init_model(num_classes)#, use_cpu, pretrained)
    if C.LOAD_CHECKPOINT_PATH != "":
        epoch = load_model(model, optimisers, C.LOAD_CHECKPOINT_PATH)  
    logging.info("Done")
    
    logging.info("Start training")
    model = train_model(model, dataloaders, datasets, optimisers, criterion, epoch, device)
    logging.info("Finished, saving...")
    save_model(model, optimisers, C.EPOCHS, C.CHECKPOINT_PATH + datetime.datetime.today().strftime('%Y-%m-%d') + "/FINISHED-")
    logging.info("Exiting")

if __name__=="__main__":
    main()