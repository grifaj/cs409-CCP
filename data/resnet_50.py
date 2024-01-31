import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import csv
import pandas as pd
# import traceback
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
# from torchsummary import summary


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
  

def init_dataset(data_file, test_size=0.25):
    '''
        Initializes dataset and dataloaders used for training model. Reads data from CSV file <data_file>. Splits data into train, validation and test sets.
    '''
    if not os.path.exists(data_file):
        print('[INFO] Data csv does not exist.')
        return False, None, None, None, None
    else:
        data = pd.read_csv(data_file, names=['file_path', 'label'])
        print(data)
        x = data['file_path']
        y = data['label']
        num_classes = np.unique(y)[-1]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=4)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=4)


        transformation = transforms.Compose([
                transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
                transforms.ToTensor(),
        #         transforms.Normalize(mean=MEAN, std=STD),
            ])

        train_data_object = CharactersDataSet(x_train, y_train, transformation)
        val_data_object = CharactersDataSet(x_val, y_val, transformation)
        test_data_object = CharactersDataSet(x_test, y_test, transformation)

        train_loader = torch.utils.data.DataLoader(train_data_object,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data_object,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data_object,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)
        
        dataloaders = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
        datasets = {'train': train_data_object, 'validation': val_data_object, 'test': test_data_object}

    return True, dataloaders, datasets, num_classes, len(x)


def init_model(num_classes, pretrained=False):
    '''
    Initialize device for cuda and load resnet50 model.
    '''
    ## Initialize device for cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if use_cpu:
        device = torch.device("cpu")
    
    ## Retrieve Resnet50 model
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    else:
        model = models.resnet50().to(device)
        
    for param in model.parameters():
        param.requires_grad = False  
        
    if device.type == "cuda":
        model.cuda()

    ## Initialize model
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

    # numFcInputs = model.fc[0].in_features

    model.fc = nn.Sequential(
                nn.Linear(2048, 1600),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1600, num_classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    return device, model, criterion, optimizer



## Train model
def train_model(model, criterion, optimizer, training_results_dir, num_examples, num_epochs=12):
    file_name = f'{MODEL_NAME}_B{BATCH_SIZE}_E{EPOCHS}_I{IMAGE_SIZE}_N{num_examples}.txt' # make file to store training results
    print(f'[INFO] New file created: {file_name}')
    with open(os.path.join(training_results_dir, file_name), 'w') as f:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            f.write('Epoch {}/{}\n'.format(epoch+1, num_epochs))
            print('-' * 10)
            f.write('-' * 10+'\n')

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / datasets[phase].__len__()
                epoch_acc = running_corrects.double() / datasets[phase].__len__()

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                f.write('{} loss: {:.4f}, acc: {:.4f}\n'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
    f.close()
    
    return model

def gen_csv_file(data_file, data_dir, file_ext, log_file):
    with open(data_file, 'w', newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        for (root,dirs,files) in os.walk(data_dir, topdown=True):
            print(root)
            if len(root.split("/")[1]) == 0: # the root directory, no images here so skip
                continue
            label = root.split("/")[1] # sub-directory names are class labels

            for file in glob.glob(root + '/*' + file_ext):
                
                try: # check that image can be opened - add additional image requirement checks here
                    img = Image.open(file).convert("L")
                    im = np.asarray(img)
                    csv_writer.writerow([file, label])
                    print(file)
                except Exception as e:
                    with open(os.path.join(data_dir, log_file), 'a') as f:
                        f.write(str(e)+"\n")
    #                 print(str(traceback.format_exc()))
                
        csv_file.close()


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("BATCH_SIZE", type=int)
    parser.add_argument("EPOCHS", type=int)
    parser.add_argument("IMAGE_SIZE", type=int)
    parser.add_argument("MODEL_NAME", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("file_ext", type=str)
    parser.add_argument("test_size", type=float)
    parser.add_argument("pretrained", type=bool)
    parser.add_argument("data_file", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("results_dir", type=str)
    parser.add_argument("log_file", type=str)
    parser.add_argument("use_cpu", type=bool)

    args = parser.parse_args()

    ## Initialise model parameters
    BATCH_SIZE = args.BATCH_SIZE
    EPOCHS = args.EPOCHS
    IMAGE_SIZE = args.IMAGE_SIZE
    MODEL_NAME = args.MODEL_NAME
    data_dir = args.data_dir
    file_ext = args.file_ext
    test_size = args.test_size
    pretrained = args.pretrained
    data_file = args.data_file
    model_path = args.model_path
    results_dir = args.results_dir
    log_file = args.log_file
    use_cpu = args.use_cpu

    # if gen_csv:
    #     gen_csv_file(data_file, data_dir, file_ext, log_file)

    found, dataloaders, datasets, num_classes, num_examples = init_dataset(data_file, test_size)
    
    if not found:
        print(f'[INFO] Dataset could not be loaded.')
    else:     
        device, model, criterion, optimizer = init_model(num_classes, pretrained)
        model = train_model(model, criterion, optimizer, results_dir, num_examples, num_epochs=EPOCHS)
        torch.save(model.state_dict(), model_path)