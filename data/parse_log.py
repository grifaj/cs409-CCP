import os
import matplotlib.pyplot as plt
import re 
import argparse
import numpy as np

LOG_PATH = 'run.log' # Path to log containing all job logs (master log file)
OUTPUT_DIR = '/dcs/project/seal-script-project-data/job_logs/logs' # Location to store job logs
PLOT_DIR = '/dcs/project/seal-script-project-data/job_logs/plots' # Location to store convergence plots

def plot_convergence(log):


    train = {'loss': [], 'acc': []}
    validation = {'loss': [], 'acc': []}
    i = 0
    with open(log, 'r') as f:
        while True:
            line = f.readline()
            if i == 0: # Create filename for plots 
                filename = line[line.index('['):line.index(']')+1]
            if re.search('Epoch \d+/\d+', line) is not None: # There is a match
                loss = re.search('Loss \d+.\d{1,5}', line)
                acc = re.search('Accuracy \d+.\d{1,5}', line)
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
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, filename + '_loss.png'))

    plt.clf()

    plt.plot(np.arange(1,len(acc_t)+1), acc_t, linewidth=1, c='red', label='Train')
    plt.plot(np.arange(1,len(acc_v)+1), acc_v, linewidth=1, c='blue', label='Validation')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, filename + '_acc.png'))

    print(f'[INFO] Plots saved to {PLOT_DIR}')

def split_logs():
    with open(LOG_PATH, 'r') as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true') # Include if master log file should be split into separate job log files
    parser.add_argument('--plot', action='store_true') # include if job log file should be plotted
    parser.add_argument('--job_log', type=str) # File of job log to plot (should be in OUTPUT_DIR)
    
    args = parser.parse_args()

    if args.split:
        split_logs() # Split log file into different job logs

    if args.plot:
        plot_convergence(args.job_log) # Create convergence plots