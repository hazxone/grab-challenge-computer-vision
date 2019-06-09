import scipy.io as spio
import csv
from func_utils import check_folder
import os

def main(args = None):
    # parse arguments
    args = parse_args(args)

    # Separate the process into train or test
    # If the user pass --test parameter, it will process the test files
    if args.test:
        head, tail = os.path.split(args.test-csvfile)
        check_folder(head)
        mat = spio.loadmat(args.test-matfile, squeeze_me=True)
        open_csv = open(args.test-csvfile, mode="w", newline="")
    else:
        head, tail = os.path.split(args.csvfile)
        check_folder(head)
        mat = spio.loadmat(args.matfile, squeeze_me=True)
        open_csv = open(args.csvfile, mode="w", newline="")

    # Load .mat file into array
    row_mat = mat['annotations']

    # Open new csv file for writing
    write_csv = csv.writer(
        open_csv, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )

    # Loop each row in the opened mat and write it to csv
    # Car class is padded with zeros to it will be in the right order when convert it to dataframe during training
    for row in row_mat:
        x1, y1, x2, y2, car_id, filepath = row
        padded_car_id = '{0:03d}'.format(int(car_id))
        write_csv.writerow([filepath, x1, y1, x2, y2, padded_car_id])

    open_csv.close()


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Parse arguments.')

    parser.add_argument('--matfile', default='dataframe/mat_files/cars_train_annos.mat', help='Path to the mat file.')
    parser.add_argument('--csvfile', default='dataframe/csv_files/cars_train.csv', help='Path to save the csv.')
    parser.add_argument('--test', action='store_true', help='Process test mat file')
    parser.add_argument('--test-matfile', default='dataframe/mat_files/cars_test_annos_withlabels.mat', help='Path to the mat file.')
    parser.add_argument('--test-csvfile', default='dataframe/csv_files/cars_test.csv', help='Path to save the csv.')
    
    return parser.parse_args(args)

if __name__ == "__main__":
    main(args=None)