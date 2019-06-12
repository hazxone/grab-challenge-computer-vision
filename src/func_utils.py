import csv
import os
import numpy as np
from keras.models import load_model
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image

def read_from_csv(csv_file_name, number_of_columns):
    """Read from csv and convert it to python array
    
    Arguments:
        csv_file_name {str} -- Path to csv file
        number_of_columns {int} -- Number of columns in the csv

    Returns:
        array -- With size columns X rows

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
    """Load keras model
    
    Arguments:
        model_name {str} -- Path to weight file (.h5)

    Returns:
        model -- Keras model with trained weights
    """
    model = load_model(model_name)
    return model

def image_to_tensor(file_path, image_size=224):
    """Convert image file to 4D tensor to feed into model.predict
    
    Arguments:
        file_path {str} -- Path to image file
        image_size {int} -- Image size to be resize. DenseNet image size is 224 px (default: {224})
    
    Returns:
        4D Tensor -- Image file that has been normalized / scaled and added batch dimension
    """
    img = image.load_img(file_path, target_size=(image_size, image_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def check_folder(folder_path):
    """Check if folder exist, if not create directory
    
    Arguments:
        folder_path {str} -- Path to folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)