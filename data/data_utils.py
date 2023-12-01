# File for functions/classes utilised across data collection/model training files

from character_translation_load import DatasetLoad
import os

### PARAMS
#   edit_index = the maximum character index to load into the csv
def get_data_csv_override(edit_index, data_dir):
    if os.path.exists(os.path.join(data_dir, 'trainData.csv')):
        os.remove(os.path.join(data_dir, 'trainData.csv'))
    DatasetLoad(data_dir, edit_index, 'trainData.csv').createCsv() 

### PARAMS
#   edit_index = the maximum character index to load into the csv
def get_data_csv(edit_index, data_dir):
    if not os.path.exists(os.path.join(data_dir, 'trainData.csv')):
        DatasetLoad(data_dir, edit_index, 'trainData.csv').createCsv() 