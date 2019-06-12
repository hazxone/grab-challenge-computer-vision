import argparse
import csv

import numpy as np

from func_utils import image_to_tensor, load_keras_model, read_from_csv
from sklearn.metrics import classification_report


def main(args = None):
    args = parse_args(args)

    # Read the class list and convert it into dictionary e.g. {car_class:car_name}
    class_list = read_from_csv(args.classcsv, 2)
    class_dict = {int(car[1]):car[0] for car in class_list}

    # Load the model for prediction
    model = load_keras_model(args.model)

    # Open new csv for writing false prediction
    open_csv = open(args.falsecsv, mode="w", newline="")
    write_csv = csv.writer(open_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

    true_pred = 0
    false_pred = 0
    y_preds = []
    y_test = []

    # Read the csv list of cropped files and convert it to array of [file_path, car_id]
    evaluate_list = read_from_csv(args.testcsv, 2)

    for car in evaluate_list:
        file_path = str(car[0])
        car_id = int(car[1])

        # Load, resize, convert to 4D tensor and normalize image before predict
        x = image_to_tensor(file_path, 224)
        preds = model.predict(x)

        # Get the class index of highest probability result and shift by 1 since python is zero based index
        category_preds = int(np.argmax(preds)) + 1

        # Uncomment to print out the car name and probability
        # print(class_dict[category_preds]), np.max(preds))

        # Create list of predictions and ground truth to be feed into sklearn classification_report
        y_preds.append(category_preds)
        y_test.append(car_id)

        if category_preds == car_id:
            true_pred += 1
        else:
            false_pred += 1
            write_csv.writerow([class_dict[car_id], class_dict[category_preds], np.max(preds)])

    print("Number of true prediction: ",true_pred, "Number of false prediction: ",false_pred)
    print("Accuracy: {:.4f}% ".format(100*(true_pred/(true_pred + false_pred))))

    print(classification_report(y_test, y_preds))
    open_csv.close()
    
def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Parse arguments.')

    parser.add_argument('--model', default='snapshots/DenseNet169-epochs-47-0.26.h5', help='Path to the model weight.')
    parser.add_argument('--classcsv', default='dataframe/csv_files/class.csv', help='Path to a CSV file containing class label.')  
    parser.add_argument('--testcsv', default='dataframe/csv_files/cars_test_crop.csv', help='Path to a CSV file containing test images.')
    parser.add_argument('--falsecsv', default='dataframe/csv_files/false_prediction.csv', help='Path to a CSV file containing test images.')
    return parser.parse_args(args)

if __name__ == "__main__":
    main(args=None)
