import scipy.io as spio
import csv
from func_utils import check_folder

def main(args = None):
    # parse arguments
    args = parse_args(args)

    head, tail = os.path.split(args.csvfile)
    check_folder(head)

    # Load .mat file into array
    mat = spio.loadmat(args.matfile, squeeze_me=True)
    row_mat = mat['annotations']

    # Open new csv file for writing
    open_csv = open(args.csvfile, mode="w", newline="")
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

    parser.add_argument('--matfile', default='dataframe/mat_files/cars_train_annos.met', help='Path to the mat file.')
    parser.add_argument('--csvfile', default='dataframe/csv_files/cars_train.csv', help='Path to save the csv.')
    return parser.parse_args(args)

if __name__ == "__main__":
    main(args=None)