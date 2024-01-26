# cs409-CCP

## Training model

<MODEL_PATH>: the path to save the trained model. Set as default: output/model.pth

<PLOT_PATH>: the path to save the error/loss plot generated during training. Set as default: output/plot.png

<CHAR_INDEX>: train the model on characters 1...<CHAR_INDEX>. Check hsk.csv for character indexes

- `python character_translation_train.py --model <MODEL_PATH> --index <PLOT_PATH> --index <CHAR_INDEX>`
