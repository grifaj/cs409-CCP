import os
import matplotlib.pyplot as plt
import re 
import argparse
import numpy as np

MASTER_LOG_PATH = './job_logs/run/mobilenet_v3_large_run.siamese.log' # Path to log containing all job logs (master log file)
OUTPUT_DIR = './job_logs/logs/siamese_mobilenet' # Location to store job logs
PLOT_DIR = './job_logs/plots/siamese_mobilenet' # Location to store convergence plots

JOB_LOG_PATH = './job_logs/logs/siamese_mobilenet/[2024-04-17 20:12:37].log' # Path to job log to plot graphs for

MODEL_NAME = 'Siamese-MobileNetV3'
PRETRAINED = False

def plot_convergence():
    # Create path to store plots
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    train = {'loss': [], 'acc': []}
    validation = {'loss': [], 'acc': []}
    i = 0
    with open(JOB_LOG_PATH, 'r', newline='') as f:
        while True:
            line = f.readline()
            if i == 0: # Create filename for plots 
                filename = line[line.index('['):line.index(']')+1]
            if re.search('Epoch \d+/\d+', line) is not None: # There is a match
                loss = re.search('Loss (\d+.\d{1,5})|Loss (\d+)', line)
                acc = re.search('Accuracy (\d+.\d{1,5})|Accuracy (\d+)', line)
                print(loss)
                if loss is None or acc is None:
                    break
                loss = loss.group().split(" ")[1]
                acc = acc.group().split(" ")[1]
                if 'train' in line:
                    train['loss'].append(loss)
                    train['acc'].append(acc)
                elif 'validation' in line:
                    validation['loss'].append(loss)
                    validation['acc'].append(acc)
            i+=1
            # End of file is reached
            if not line:
                break
    f.close()

    # Convert lists to numpy arrays
    loss_t = np.array(train['loss'], dtype=np.float16)
    acc_t = np.array(train['acc'], dtype=np.float16)

    loss_v = np.array(validation['loss'], dtype=np.float16)
    acc_v = np.array(validation['acc'], dtype=np.float16)

    # Plot loss and accuracy curves with train and validation
    plt.plot(np.arange(1,len(loss_t)+1), loss_t, linewidth=1, c='red', label='Train')
    plt.plot(np.arange(1,len(loss_v)+1), loss_v, linewidth=1, c='blue', label='Validation')
    plt.title(f'{"Pretrained " if PRETRAINED else ""}{MODEL_NAME} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, filename + '_loss.png'))

    plt.clf()

    plt.plot(np.arange(1,len(acc_t)+1), acc_t, linewidth=1, c='red', label='Train')
    plt.plot(np.arange(1,len(acc_v)+1), acc_v, linewidth=1, c='blue', label='Validation')
    plt.title(f'{"Pretrained " if PRETRAINED else ""}{MODEL_NAME} accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, filename + '_acc.png'))

    print(f'[INFO] Plots saved to {PLOT_DIR}')

def split_logs():
    # Create path to store job logs
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(MASTER_LOG_PATH, 'r') as f:
        while True:
            line = f.readline()
            if 'Logger started' in line:
                filename = line[line.index('['):line.index(']')+1] + '.log'
                if not os.path.exists(filename):
                    write_to = True # Flag to indicate that file was created during this script or by another script
                    with open(os.path.join(OUTPUT_DIR, filename), 'w', ) as w:
                        w.write(line)
                    w.close()
                else:
                    write_to = False
            else:
                if write_to: # Dont write to the file if it was created before this script was run
                    with open(os.path.join(OUTPUT_DIR, filename), 'a', ) as w:
                        w.write(line)
                    w.close()

            # End of file is reached
            if not line:
                break

    f.close()



if __name__=='__main__':
    # If no individual job log is set, split the master log into separate job logs
    if JOB_LOG_PATH == "":
        split_logs() # Split log file into different job logs

    # If an individual job log is set, plot that job log
    if JOB_LOG_PATH != "":
        plot_convergence() # Create convergence plots