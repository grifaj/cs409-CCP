import numpy as np
from PIL import Image
import os
import glob
import csv
import pandas as pd
# import traceback
# import argparse
from tqdm import tqdm
from config import Config_2 as C
import logging
from datetime import datetime
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split



# initialise logger
logging.basicConfig(filename=C.LOG_PATH + ".siamese.log", encoding="utf-8", level=logging.INFO, 
                    format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger().addHandler(logging.StreamHandler())
logging.info("==========Logger started===========")
## Initialize device for cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(torch.cuda.is_available())
print((torch.cuda.is_available()))
# change model download path to /large
if C.TORCH_MODEL_CACHE:
    os.environ["TORCH_HOME"] = C.TORCH_MODEL_CACHE
    
    
class SiameseDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x 
        self.y = y 
        self.transform = transform
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        """we'll just do it like that"""
        def openImage(path):
            image = Image.open(path).convert("L")
            image = Image.fromarray(np.repeat(np.asarray(image)[..., np.newaxis], 3, -1))
            if self.transform:
                image = self.transform(image)
            return image
        
        anchor = openImage(self.x.iloc[index])
        anchor_label = self.y.iloc[index]
        
        positive = openImage(np.random.choice(self.x[self.y == anchor_label]))
        
        negative = openImage(np.random.choice(self.x[self.y != anchor_label]))
        
        return anchor, positive, negative
    
    
class SiameseNetwork(nn.Module):
    def __init__(self, backbone, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        
    def forward(self, x):
        out = self.backbone(x)
        return out.view(out.size()[0], -1)
    

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, *args, **kwargs) -> None:
        """Initialise with margin. Margin is set to 0.3 in siamese paper"""
        super().__init__(*args, **kwargs)
        self.margin = margin 
        
    def forward(self, anchor, pos, neg):
        # L = max(d(a, p) − d(a, n) + m, 0)  -- relu is just max
        loss =  torch.nn.functional.relu(torch.linalg.vector_norm(anchor - pos) - torch.linalg.vector_norm(anchor - neg) + self.margin)
        # print(loss)
        return loss
    
    
def init_dataset(model_type):
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
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=C.TEST_SIZE, stratify=y)

        if model_type == 'resnet_50':
            weights = models.ResNet50_Weights.DEFAULT
        elif model_type == 'mobilenet_v3_large':
            weights = models.MobileNet_V3_Large_Weights.DEFAULT

        transformation = weights.transforms()

        train_data_object = SiameseDataset(x_train, y_train, transformation)
        val_data_object = SiameseDataset(x_val, y_val, transformation)

        train_loader = torch.utils.data.DataLoader(train_data_object,  # type:ignore
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        val_loader = torch.utils.data.DataLoader(val_data_object,  # type:ignore
                                                batch_size=C.BATCH_SIZE,
                                                shuffle=C.SHUFFLE_DATA)
        
        dataloaders = {'train': train_loader, 'validation': val_loader}
        datasets = {'train': train_data_object, 'validation': val_data_object}

    return dataloaders, datasets, num_classes
    
    
def init_model(backbone_type, pretrained=False, log=True):
    
    logging.info(f"Initialise siamese with {backbone_type} backbone")
    
    # remove last feed forward layer and make it an identity -- we want the embeddings only
    if backbone_type == "resnet_50":
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights = weights)
        else:
            backbone = models.resnet50()
        backbone.fc = nn.Identity()  # type:ignore  # 2048 features 
    elif backbone_type == 'mobilenet_v3_large':
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            backbone = models.mobilenet_v3_large(weights = weights)
        else:
            backbone = models.mobilenet_v3_large()
        backbone.classifier = nn.Identity()  # type:ignore  # 960 features 
    else:
        raise ValueError(f"Model {backbone_type} is not allowed yet")
    
            
    model = SiameseNetwork(backbone)
    criterion = TripletLoss()
    optimisers = [optim.Adam(model.parameters(), lr=C.LEARNING_RATE, betas=C.ADAM_BETA)]
    
    model.to(device)
    criterion.to(device)
        
    logging.info(f"Using device {device.type}")
    logging.info(f"Pretrained weights: {pretrained}")
                
    if log:
        logging.info(model)
    
    return model, criterion, optimisers 


## Save model dictionaries
def save_model(model:SiameseNetwork, optimisers, epoch:int):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_a": optimisers[0].state_dict()
    }, C.CHECKPOINT_PATH + f"SIAMESE-CK-{epoch}.pt")

def load_model(model:SiameseNetwork, optimisers, load_from:str):
    logging.info(f"Loading checkpoint {load_from}")
    checkpoint = torch.load(load_from)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimisers[0].load_state_dict(checkpoint["optimiser_a"])
    epoch = checkpoint["epoch"]
    logging.info("Loading done")
    
    return model, epoch
    
    
def train_model(model:SiameseNetwork, dataloaders, datasets, optimisers, criterion:TripletLoss, start_epoch=0):
    def zero_grads(optimisers):
        for o in optimisers: o.zero_grad()
        
    def steps(optimisers):
        for o in optimisers: o.step()
        
    if C.CHECKPOINT_PATH[-1] != "/": C.CHECKPOINT_PATH += "/"
    C.CHECKPOINT_PATH += datetime.today().strftime('%Y-%m-%d') + "/"
    os.makedirs(C.CHECKPOINT_PATH, exist_ok=True)
    logging.debug("Checkpoint files ok")
    
    logging.info("starting training")
    for epoch in range(start_epoch, C.EPOCHS):
        # print(f"Epoch {epoch+1}/{C.EPOCHS}")
        
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Training phase")
            else:
                model.eval()
                logging.debug(f"[Epoch {epoch+1}/{C.EPOCHS}] Validation phase")
                
            running_loss = 0.0
            
            # for anchor, pos, neg in dataloaders[phase]:
            logging.warn("!!! Using TQDM !!!")
            for anchor, pos, neg in tqdm(dataloaders[phase]):
                anchor = anchor.to(device)
                pos = pos.to(device)
                neg = neg.to(device)
                
                out_a = model(anchor)
                out_p = model(pos)
                out_n = model(neg)
                
                loss = criterion(out_a, out_p, out_n)
                
                if phase == 'train':
                    zero_grads(optimisers)
                    loss.backward()
                    steps(optimisers)
                    
                running_loss += loss.item() * anchor.size(0)
                
            epochLoss = running_loss / datasets[phase].__len__()
            
            # we do not have accuracy with triplet loss
            logging.info(f"[Epoch {epoch+1}/{C.EPOCHS}] {phase} [Loss {epochLoss:.4f}] [Accuracy 0]")
            
            if phase == "train" and (epoch + 1) % C.SAVE_EVERY_N == 0:
                    save_model(model, optimisers, epoch)
                    
    return model, optimisers, epoch 
    

def main():
    pretrained = C.PRETRAINED
    backbone_type = C.MODEL_NAME
    logging.info("Loading dataset and dataloaders")
    dataloaders, datasets, num_classes = init_dataset(backbone_type)  # type:ignore
    logging.info(f"Num classes {num_classes} (not that it matters)")
    logging.info("Done")
    epoch = 0
    logging.info(f"Loading siamese with backbone {backbone_type}")
    model, criterion, optimisers = init_model(backbone_type, pretrained)
    if C.LOAD_CHECKPOINT_PATH != "":
        model, epoch = load_model(model, optimisers, C.LOAD_CHECKPOINT_PATH)
    logging.info("Done")
    
    logging.info("Start training")
    model, optimisers, epoch = train_model(model, dataloaders, datasets, optimisers, criterion, epoch)
    logging.info("Finished, saving...")
    save_model(model, optimisers, epoch)
    logging.info("Exiting")


if __name__ == "__main__":
    main()
