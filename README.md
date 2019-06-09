# Grab Challenge - Computer Vision

This is a submission for Grab - AIforSEA Challenge. I choose the **Computer Vision - Recognizing Car Details**

![Screenshot](jupyter_notebook/test_image/grab_header.png)

## Solution Description

This is the folder layout for the whole repo.\
Raw data from dataset are organized in the **data** folder. The processing (cropping) of the raw images are stored under **data/crop_images**.\
All meta data such as mat files and csv files are under **dataframe** folder.

```
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
|       ├── 00001.jpg
|       ├── ....
|       └── 08144.jpg
|   └── car_test
|       ├── 00001.jpg
|       ├── ....
|       └── 08041.jpg
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
│   ├── Example.ipynb
|   └── test_image
|
└── README.md
```

## Data Processing

### First Step - Convert .mat to csv
First we need to convert the mat  file **car_train_annos.mat** to csv.
```bash
python3 src/mat_to_csv.py
```
The output file is **car_train.csv** in the 'dataframe/csv_files'

> ### Why I use CSV
> * I focused to use the input of training and evaluation in CSV file format.
> * CSV file is much easier to create, access, manipulate, convert to PandasDataframe and it is in general better compability format compared to Matlab file.
> * Hence, the reason I've separate the process of converting mat file to csv file from the data clean up (croppping the image).

### Second Step - Crop Car Images
Then we need to process the raw images.\
This script will crop the car images in 'data/car_train' according to the bounding box, save it in the 'data/crop_images'.\
It read the file and bounding box from the csv file created above (**car_train.csv**).\
It will then create csv file **car_train_crop.csv** with structure "file_name, car_id" in 'dataframe/csv_files'.

```bash
python3 src/data_preprocessing.py
```
![Screenshot](jupyter_notebook/test_image/crop_image.png)

## Training

1. I've tried the training with various backbone architecture such as Resnet, ResnetV2, InceptionResnetV2.
In the end, DenseNet169 gave the lowest validation error. Not only that, DenseNet has much lower trainable parameters (12M) compared
to the same layers in ResNet (60M).

![Screenshot](jupyter_notebook/test_image/cometml.png)

```bash
python3 src/train_densenet.py
```

2. Using `model.fit` to fit the whole train dataset in the memory will not be viable since we got more than 8000+ images and it is not scalable to train very large data in the future.
3. Since I already have data the CSV list, I just need to convert it to Pandas Dataframe to pass it to Image Generator via `flow_from_dataframe`. This way,
the images will be generated on the fly batch by batch as the training process run.
4. Initially, using `flow_from_dataframe`, the output class were totally out of order since it sort the class by 1,10,100,101.. instead of 1,2,3,4..
I've managed to overcome this by padding the class number with zeros 001,002,003 when converting from mat file to csv (mat_to_csv.py)
5. `flow_from_dataframe` also better with this dataset than `flow_from_directory` since we don't have to rearrange the folders back according to the classes.

### Result
After 50 epochs, I got validation loss of 0.22

![Screenshot](jupyter_notebook/test_image/val_loss.png)

## Testing / Evaluation

1. After training, we can use the test dataset (8041 images). First we need to convert the mat file to csv. Extract the test images from tar.gzip in the 'data/car_test', and copy **cars_test_annos_withlabels.mat** to 'dataframe/mat_files'

```bash
python3 src/mat_to_csv.py --test
```

⋅⋅⋅The output will be **cars_test.csv** in the 'dataframe/csv_files' folder⋅⋅

2. Then we need to crop the test images according to the bounding box

```bash
python3 src/data_preprocessing.py --test
```
⋅⋅⋅The output will be **cars_test_crop.csv** in the 'dataframe/csv_files' folder and cropped images in 'data/crop_images/test'⋅⋅

3. Run the test evaluation

⋅⋅⋅The pre-trained weight can be download here⋅⋅

```bash
python3 src/evaluate.py --model snapshots/densenet.h5 --testcsv dataframe/csv_files/car_test_crop.csv --classcsv dataframe/csv_files/classes.csv
```

### Result

I got **90.8%** Accuracy (7303 true predictions out of 8041 images)

![Screenshot](jupyter_notebook/test_image/evaluate.png)