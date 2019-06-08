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
|   ├── car_train
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
## Data Analysis

First we need to process the raw images.
This script will crop the car images in 'data/car_train' according to the bounding box, save it in the 'data/crop_images'
It will then create csv file (car_train_crop.csv) with structure "file_name, car_id" in 'data/csv_files'

```bash
python3 src/data_preprocessing.py

# Optional : Pass argument --test to process test images (car_test)
python3 src/data_preprocessing.py --test
```

### Training

![Screenshot](jupyter_notebook/test_image/screenshot.png)

### Testing

## Installation

## Evaluate

```bash
python3 src/evaluate.py --weight snapshots/densenet.h5 --classlist data/csv_files/class.csv
```
