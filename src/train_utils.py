import csv
from keras.models import load_model
from keras.applications.densenet import preprocess_input

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
        Array of cols X rows

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

def load_keras_model(model_name):
    model = load_model(model_name)
    return model

def image_to_tensor(file_path, image_size=224):
    img = image.load_img(file_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x