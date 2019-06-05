import csv
import cv2
import train_utils
import argparse

def main( mode_process = 'train' ):
    # Open csv list to crop all images wrt to its bounding box
    # Csv is in format [file_name, x1, y1, x2, y2, car_id]
    cars_database = read_from_csv("devkit/car_{}.csv".format(mode_process), 6)

    # Make new csv to write back new list of the cropped images and the car class (car_id)
    open_csv = open("devkit/cars_{}_crop.csv".format(mode_process), mode="w", newline="")
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
        new_file_path = "cars_{}_crop/".format(mode_process) + split_name[1]
        image = cv2.imread(car[0])
        crop_image = image[y1 : y1 + (y2 - y1), x1 : x1 + (x2 - x1)]
        cv2.imwrite(new_file_path, crop_image)
        write_csv.writerow([new_file_path, car[5]])

    open_csv.close()

    # Preprocessing Done
    # Right now we have:
    # 1. New folders (cars_train_crop) contains all the cropped image according to bounding box
    # 2. New csv file (cars_train_crop.csv) that list the cropped images and its correspond class : [file_path, car_id]


parser = argparse.ArgumentParser(description='passing argument to process either train or test set')

parser.add_argument('--test', type=bool, default=False,
                    help='To use test set, default is train set')

args = parser.parse_args()

if args.test:
    mode_process = "test"

if __name__ == "__main__":
    main( mode_process = mode_process )
