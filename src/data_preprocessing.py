import csv
import cv2
from train_utils import read_from_csv

# Open csv list to crop all images wrt to its bounding box
# Csv is in format [file_name, x1, y1, x2, y2, car_id]
cars_database = read_from_csv("devkit/car_train.csv", 6)

# Make new csv to write back new list of the cropped images and the car class (car_id)
open_csv = open("devkit/cars_train_crop.csv", mode="w", newline="")
write_csv = csv.writer(
    open_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
)

# Loop each row; crop, save and write the result to cars_train_crop.csv
for car in cars_database:
    x1 = int(car[1])
    y1 = int(car[2])
    x2 = int(car[3])
    y2 = int(car[4])
    split_name = car[0].split("/")
    new_file_path = "cars_train_crop/" + split_name[1]
    image = cv2.imread(car[0])
    crop_image = image[y1 : y1 + (y2 - y1), x1 : x1 + (x2 - x1)]
    cv2.imwrite(new_file_path, crop_image)
    write_csv.writerow([new_file_path, car[5]])

# Preprocessing Done
# Right now we have:
# 1. New folders (cars_train_crop) contains all the cropped image according to bounding box
# 2. New csv file (cars_train_crop.csv) that list the cropped images and its correspond class : [file_path, car_id]
