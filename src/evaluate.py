import numpy as np
import csv
from sklearn.metrics import classification_report
import argparse
from func_utils import read_from_csv, load_keras_model, image_to_tensor

def main(args = None):
    # parse arguments
    args = parse_args(args)

    # Read the class list and convert it into dictionary 
    class_list = read_from_csv(args.classcsv, 2)
    class_dict = {int(car[1]):car[0] for car in class_list}

    # Load the model for prediction
    model = load_keras_model(args.model)

    true_pred = 0
    false_pred = 0
    y_preds = []
    y_test = []

    # Read the csv cropped file and convert it to array of [file_path, car_id]
    evaluate_list = read_from_csv(args.testcsv, 2)

    for car in evaluate_list:
        file_path = str(car[0])
        car_id = int(car[1])

        x = image_to_tensor(file_path, 224)
        preds = model.predict(x)

        # Get the class index of highest probability result
        category_preds = int(np.argmax(preds)) + 1

        # Print out the car name and probability
        # print(class_dict[category_preds]), np.max(preds))

        # Create list of predictions and ground truth to be feed into sklearn classification_report
        y_preds.append(category_preds)
        y_test.append(car_id)

        if category_preds == car_id:
            true_pred += 1
        else:
            false_pred += 1

    print("Number of true prediction: ",true_pred, "Number of false prediction: ",false_pred)
    print("Accuracy: ",(100*(true_pred/(true_pred + false_pred))),"%")

    print(classification_report(y_test, y_preds))

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Parse arguments.')

    parser.add_argument('--model', default='snapshots/densenet.h5',, help='Path to the model weight.')
    parser.add_argument('--classcsv', default='dataframe/csv_files/class.csv', help='Path to a CSV file containing class label.')  
    parser.add_argument('--testcsv', default='dataframe/csv_files/cars_test_crop.csv', help='Path to a CSV file containing test images.')
    return parser.parse_args(args)

if __name__ == "__main__":
    main(args=None)