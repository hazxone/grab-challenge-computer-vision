import csv

def read_from_csv(csv_file_name, number_of_columns):
    """Read from csv and convert to array.

    Parameters
    ----------
    csv_file_name : string
        Path to csv file
    number_of_columns : int
        How many columns in the csv

    Returns
    -------
    array
        Array of cols x rows

    """
    full_array = []
    with open(csv_file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            temp_array = []
            for i in range(number_of_columns):
                temp_array.append(row[i])
            full_array.append(temp_array)
    return full_array
