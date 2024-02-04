import os
from PIL import Image
import csv
import glob
import numpy as np
import argparse

## Create log used for reporting issues
## Create training log
def create_log_files(data_dir, results_dir, log_file):
    '''
    Creates a text file for logging issues with dataset loading during training under <data_dir>/<log_file>.
    Also creates a new directory <log_dir> for storing model training results.
    '''
    with open(log_file, 'w') as f:
        f.close()
        
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


## Generate dataset CSV
def generate_dataset_csv(data_dir, data_file, file_ext, log_file):
    '''
    Generates CSV file used by CNN model. CSV is named <data_file>. Is overwritten if already exists.
    Contains list of image paths and their labels.
    CSV is created by reading all files with <file_ext> extensions from subdirectories in <data_dir> where label is the name of the subdirectory.
    Issues with loading images are written to a log at <log_dir> held under <data_dir>
    '''
    
    print(os.walk('seal-script-full/seal-script-images/'))
    with open(data_file, 'w', newline="") as csv_file:
        print(os.getcwd())
        csv_writer = csv.writer(csv_file, delimiter=",")
        for (root,dirs,files) in os.walk(data_dir, topdown=True):
            print(root)
            if len(root.split("/")[1]) == 0: # the root directory, no images here so skip
                continue
            label = root.split("/")[-1] # sub-directory names are class labels

            for file in glob.glob(root + '/*' + file_ext):
                
                try: # check that image can be opened - add additional image requirement checks here
                    img = Image.open(file).convert("L")
                    im = np.asarray(img)
                    csv_writer.writerow([file, label])
                    print(file)
                except Exception as e:
                    with open(log_file, 'a') as f:
                        f.write(str(e)+"\n")
                        f.close()
                
        csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str)
    parser.add_argument("results_dir", type=str)
    parser.add_argument("log_file", type=str)
    parser.add_argument("data_dir", type=str)
    parser.add_argument("file_ext", type=str)
    args = parser.parse_args()

    create_log_files(args.data_dir, args.results_dir, args.log_file)
    generate_dataset_csv(args.data_dir, args.data_file, args.file_ext, args.log_file)

    print(f'[INFO] Finished setup of required training files and directories.')