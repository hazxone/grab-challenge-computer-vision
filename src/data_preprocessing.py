import argparse
import csv
import os
import cv2
from func_utils import check_folder, read_from_csv
from tqdm import tqdm


def main(args = None):
    args = parse_args(args)

    if args.test:
        mode_process = "test"
    else:
        mode_process = "train"

    # Create the crop_images folder to store all the cropped images
    check_folder(os.path.join('data','crop_images'))
    check_folder(os.path.join('data','crop_images', mode_process))
    
    # Open csv list to crop all images wrt to its bounding box
    # Csv is in format [file_name, x1, y1, x2, y2, car_id]
    cars_database = read_from_csv(os.path.join('dataframe','csv_files', 'cars_{}.csv'.format(mode_process)), 6)

    # Create new csv to write new list of the cropped images's path and the car class (file_path, car_id)
    open_csv = open(os.path.join('dataframe','csv_files', 'cars_{}_crop.csv'.format(mode_process)), mode="w", newline="")
    write_csv = csv.writer(
        open_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    # Loop each row; crop, save and write the result to cars_train_crop.csv / cars_test_crop.csv
    for car in tqdm(cars_database):
        x1 = int(car[1])
        y1 = int(car[2])
        x2 = int(car[3])
        y2 = int(car[4])
        file_path = car[0]
        new_file_path = os.path.join('data','crop_images', mode_process, file_path)
        image = cv2.imread(os.path.join('data','cars_{}'.format(mode_process),car[0]))
        crop_image = image[y1 : y1 + (y2 - y1), x1 : x1 + (x2 - x1)]
        cv2.imwrite(new_file_path, crop_image)
        write_csv.writerow([new_file_path, car[5]])

    open_csv.close()

    # Preprocessing Done
    # Right now we have:
    # 1. New folders (crop_images/train) contains all the cropped image according to its respective bounding box
    # 2. New csv file (dataframe/csv_files/cars_train_crop.csv) that list the cropped images and its correspond class : [file_path, car_id]

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Parse arguments.')
    parser.add_argument('--test', action='store_true', help='Change mode to process test set')
    
    return parser.parse_args(args)

if __name__ == "__main__":
    main(args=None)
