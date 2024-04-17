class Config:
    # path the python logger will store files to
    LOG_PATH = "./run.log"
    # path to read file names from (needs to be in a train.csv format)
    DATA_PATH = "./train.csv"
    # path which pretrained torch weights are stored to
    TORCH_MODEL_CACHE = "/dcs/large/u2009169/torch-cache/"
    
    # batch size
    BATCH_SIZE = 4
    # shuffle order of data in torch data loader
    SHUFFLE_DATA = True 
    
    # learning rate
    LEARNING_RATE = 0.0001
    # beta_1 and beta_2 momentum parameters for Adam
    ADAM_BETA = (0.5, 0.99)
    
    # total training epochs
    EPOCHS = 200
    # path to save checkpoints
    CHECKPOINT_PATH = "/dcs/large/seal-script-project-checkpoints/vgg19/"
    # save checkpoint every n epochs
    SAVE_EVERY_N = 20
    # load model checkpoint path -- leave blank to not load
    LOAD_CHECKPOINT_PATH = ""

class Config_2:
    model_types = ['resnet_50', 'mobilenet_v3_large', 'vgg_19']

    # model name - select architecture to train
    MODEL_NAME = model_types[1]
    # path the python logger will store files to
    LOG_PATH = f"./job_logs/run/{MODEL_NAME}_run.log"
    # path to read file names from (needs to be in a train.csv format)
    DATA_PATH = "./trainData.csv"
    # path which pretrained torch weights are stored to
    TORCH_MODEL_CACHE = ""
    
    # batch size
    BATCH_SIZE = 18
    # shuffle order of data in torch data loader
    SHUFFLE_DATA = True 
    # proportion of data for testing
    TEST_SIZE = 0.3
    # size of square images
    IMAGE_SIZE = 128
    # use the pretrained weights for chosen model
    PRETRAINED = False
    
    # learning rate
    LEARNING_RATE = 0.00001
    # beta_1 and beta_2 momentum parameters for Adam
    ADAM_BETA = (0.5, 0.99)
    
    # total training epochs
    EPOCHS = 100
    # path to save checkpoints
    CHECKPOINT_PATH = f"/dcs/large/seal-script-project-checkpoints/{MODEL_NAME}{'_pretrained' if PRETRAINED else ''}/"
    # save checkpoint every n epochs
    SAVE_EVERY_N = 20
    # load model checkpoint path -- leave blank to not load
    LOAD_CHECKPOINT_PATH = ""

    # path of checkpoint to convert
    CONVERT_CHECKPOINT_PATH = "/dcs/large/seal-script-project-checkpoints/mobilenet_v3_large/2024-04-13/CK-59.pt"
    # path to store ONNX models
    BUILD_PATH = './build'
    # name of converted ONNX model
    ONNX_MODEL_NAME = f"{MODEL_NAME}_3"