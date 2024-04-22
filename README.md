# cs409-CCP

## Training model

All scripts and data used for training the Translation Model are kept under `/data/`.

### Initial setup of environment

Several files and directories are required for training the model, so a script is called prior to training.

- Run command `python init_setup.py <data_file> <results_dir> <log_file> <data_dir> <file_ext>`

`<data_file>` is the name of the file containing image paths and labels for model training.

`<results_dir>` is a directory which is created to store training results.

`<log_file>` is the name of a text file which will record issues encountered while loading and processing the dataset for training. Text file will be stored at `<data_dir>/<log_file>`.

`<data_dir>` is the directory containing the training image subdirectories.

`<file_ext>` is the file format of the training images, current is `.png`.

### Using the Resnet50 training script

Running the following command will begin training the CNN.

- `python resnet_50.py <BATCH_SIZE> <EPOCHS> <IMAGE_SIZE> <MODEL_NAME> <data_dir> <file_ext> <test_size> <data_file> <model_path> <results_dir> <log_file> -pretrained -use_cpu`

`<BATCH_SIZE>` - the batch size to use for training (integer).

`<EPOCHS>` - the number of epochs to use for training (integer).

`<IMAGE_SIZE>` - the dimensions of the (square) input images (integer).

`<MODEL_NAME>` - the name of the model for use when creating result files, current is resnet50 (string).

`<data_dir>` - the directory containing the training image subdirectories (string).

`<file_ext>` - the file format of the training images, current is `.png` (string).

`<test_size>` - the proportion of images to use for testing and validation (float). Images are split into train, test and validation by splitting twice with `<test_size>`.

`<data_file>` - the name of the csv containing image paths and labels (string).

`<model_path>` - the file where the trained model will be saved (string). Filename must end in `.pth` extension.

`<results_dir>` - the path to the directory where the training results will be recorded (string).

`<log_file>` - the name of the text file where training errors will be recorded (string).

`-pretrained` - include if the model should be trained using the pretrained model weights (bool). `False` forces the model to be trained from scratch.

`-use_cpu` - include if the training process should utilise the CPU only (bool). Set to `False` if CUDA/GPU functionality is available and preferred.
