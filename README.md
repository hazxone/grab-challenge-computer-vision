# Grab Challenge - Computer Vision

This is a submission for Grab - AIforSEA Challenge. I choose the **Computer Vision - Recognizing Car Details**

## Solution Description
```bash
root
.
├── src
│   ├── mat_to_csv.py
│   ├── data_preprocessing.py
│   ├── train_utils.py
│   ├── train_densenet.py
│   └── evaluate.py
|
├── data
|   └── car_train
|       ├── 0001.jpg
|       └── ... 8000.jpg
|   ├── car_test
|   └── crop_images (created after running data_preprocessing.py)
│       ├── train
│       └── test
|
├── dataframe
│   ├── csv_files (created after running mat_to_csv.py)
│   └── mat_files
|
├── snapshots
│   └── model_weight_file.h5 (Placed pre-trained model weight here)
|
├── jupyter_notebook
│   ├── model_weight_file.h5 (Placed pre-trained model weight here)
|   └── test_image
|
└── README.md
```
## Data Processing

### First Step
First we need to convert the mat (car_train_annos.mat) files to csv.
```bash
python3 src/mat_to_csv.py

# Optional : Pass argument --test to process test images (car_test)
python3 src/mat_to_csv.py --test
```
The output file is car_train.csv in the 'dataframe/csv_files'

### Why I use CSV
I focused to use the input of training and evaluation in CSV file.
CSV file is much easier to create, access, manipulate, convert to PandasDataframe and it is more general format compared to Matlab file.
Hence, the reason I've separate the process of converting mat file to csv file from the data clean up (croppping the image).

### Second Step
Second we need to process the raw images.
This script will crop the car images in 'data/car_train' according to the bounding box, save it in the 'data/crop_images'
It read the file and bounding box from the csv file created above (car_train.csv)
It will then create csv file (car_train_crop.csv) with structure "file_name, car_id" in 'dataframe/csv_files'

```bash
python3 src/data_preprocessing.py

# Optional : Pass argument --test to process test images (car_test)
python3 src/data_preprocessing.py --test
```

### Training

I've tried the training with various backbone architecture such as Resnet, ResnetV2, InceptionResnetV2
In the end, DenseNet169 gave the lowest error. Not only that, DenseNet has much lower trainable parameters (12M) compared
to the same layers in ResNet (60M).

![Screenshot](jupyter_notebook/test_image/screenshot.png)

```bash
python3 src/train_densenet.py
```

Using model.fit to fit the whole train dataset in the memory will not be viable since we got more than 8000+ images and it is not scalable to train very large data in the future.
Since I already have data the CSV list, I just need to convert it to Pandas Dataframe to pass it to Image Generator via flow_from_dataframe. This way,
the images will be generated on the fly batch by batch as the training process run.
Initially, using flow_from_dataframe, the output class were totally out of order since it sort the class by 1,10,100,101.. instead of 1,2,3,4..
I've managed to overcome this by padding the class number with zeros 001,002,003 when converting from mat file to csv (mat_to_csv.py)

### Testing

## Installation

## Evaluate

```bash
python3 src/evaluate.py --weight snapshots/densenet.h5 --classlist data/csv_files/class.csv
```
