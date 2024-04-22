# cs409-CCP

![alt text](https://github.com/grifaj/cs409-CCP/blob/main/images/system_overview_landscape.jpg?raw=true)

## Character classification model

The character classification model handles translation of the images of individual seal script characters into modern Chinese. 
The current version of the model uses MobileNetV3-Large[^1]. All models are implemented with PyTorch. 
All scripts and data used for training the classification model are kept under `/data`.

### Initial setup of environment

#### Generating the dataset

The current list of characters used in the dataset can be found at `/data/classical_chars.csv` and uses the first 1000 characters taken from the list of most frequently used classical Chinese characters[^2]. The format of the CSV is *class label*, *character*.

The `main` function of `/data/scrape_seals.ipynb` can be run to automatically scrape images of seal script characters. The function currently reads a spreadsheet file downloaded from[^2] and then formats the desired list of characters to a CSV, then sets up the folder structure for storing the scraped images and then searches through two sources to find any available images. As web-scraping is quite a specific task, this function may need to be altered to fit the format of your input character list. Alternatively, you can create your own dataset CSV in the format specified above, and then comment out the first four commands in the `main` function. 
The *dataDir* variable must be set, which is the location to store any scraped images. Images for a class will be stored in a directory under *dataDir* whose name will correspond to the class label. For example, if you have class labels 1,2,3,...,1000, *dataDir* will have 1000 sub-directories named 1,2,3,...,1000.

It may not be possible to scrape images for all of the characters in the dataset CSV, so some may have to be drawn by hand or found through manual searching. The `checkMissingFolders()` function in the `/data/check_data_integrity.ipynb` file can be used to find folders without any images.

The size of the dataset can be enlarged through creating distorted versions of the scraped images. To do this, use the `createCsv()` function in the `/data/create_dataset_csv.ipynb` file, first setting *DATA_DIR* to the *dataDir* location. This will create a CSV in format *image path*, *class label*. The current version of this CSV is given as `rawImages.csv` in `/examples`. Next, edit the variables in the `distort_image_batch.py` file and run it. This will create a desired number of images for each character and constitutes the final dataset.

#### Training a model

Now that a dataset has been created, you can start training a model. Training uses a CSV to find images and get their class label, so use the `createCsv()` function in the `/data/create_dataset_csv.ipynb` file again, changing the *DATA_DIR* to the root directory of the dataset.

The current dataset CSV is given as `trainData.csv` in `/examples` and contains 100 images for each class.

To train a new model, it must be added to the *model_types* list of the *Config* class in `/data/config.py`. Select the model you wish to train with *MODEL_NAME* and change the other variables (incl. model parameters, checkpoint save locations etc.) as desired. Then, in `/data/train.py`, set the dataset transformation of the new model in the `init_dataset()` function and add the model to the `init_model()` function in the same style as the three existing models. Run the `main()` function to begin training.

#### Convert model to NCNN for implementation

The process of converting a PyTorch model to NCNN format for application implementation is as follows:

PyTorch *.pt* file --> *.onnx* file --> NCNN *.bin* + *.param* files

Converting from *.pt* to *.onnx* is done by running the `/data/convert.py` file. Don't forget to set the conversion variables (_CONVERT_CHECKPOINT_PATH_, _BUILD_PATH_ and _ONNX_MODEL_NAME_) in `/data/config.py'.

For converting *.onnx* to NCNN files, we use an external tool https://convertmodel.com/. Both the *.bin* and *.param* files are necessary for implementation into the application.
## References

[^1]: Andrew Howard et al. ‘Searching for MobileNetV3’. In: CoRRabs/1905.02244 (2019). arXiv: 1905.02244. URL: http://arxiv.org/abs/1905.02244.

[^2]: Jun Da. Character frequency lists 汉字单字频率列表. https://lingua.mtsu.edu/chinese-computing/statistics/. Accessed 15th April 2024, Page last updated 最近更新: 2005-12-21
