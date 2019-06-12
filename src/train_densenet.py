import os
import pandas as pd

from func_utils import check_folder, read_from_csv

from keras import optimizers
from keras.applications.densenet import DenseNet169, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

check_folder('snapshots')

def main():
    epochs = 80
    batch_size = 29
    image_size = 224
    classes = 196

    # Read from the cropped csv and convert it to pandas Dataframe
    # since we want to use flow_from_dataframe during training
    # Class name is a 3 digits number. e.g. 001, 002 ... 196
    cars_df = pd.DataFrame(
        read_from_csv(os.path.join('dataframe','csv_files', 'cars_train_crop.csv'), 2), columns=['filename', 'class']
    )

    # Split to 7000+/1000+
    train_df, val_df = train_test_split(cars_df, test_size=0.120)

    # Construct DenseNet169 Model from keras.applications
    # Include top is set to false to make sure we can use pre-trained weights on imagenet
    # but still have custom output layers to 196 classes
    # Use Global_average_pooling as logits
    base_model = DenseNet169(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
        classes=classes,
    )
    x = base_model.output
    output_layer = Dense(classes, activation="softmax", name="fc")(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    print(model.summary())

    # Using ImageDataGenerator to augment images
    # Use padding, rotate and zoom to get high accuracy and low error
    # Referece : https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=17.0,
        width_shift_range=0.18,
        height_shift_range=0.18,
        zoom_range=0.32,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Using SGD for optimizers. Ref: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # "Fine-tuning should be done with a very slow learning rate, 
    # and typically with the SGD optimizer rather than an adaptative learning rate optimizer such as RMSProp."
    # "This is to make sure that the magnitude of the updates stays very small, so as not to wreck the previously learned features."
    sgd = optimizers.SGD(lr=1e-2, decay=5e-7, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    # Declare callbacks - Save model and reduce learning rate
    filepath = os.path.join("snapshots", "DenseNet169-epochs-{epoch:02d}-{val_loss:.2f}.h5")
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_loss", verbose=1, save_best_only=True
    )
    reduce_lr = ReduceLROnPlateau("val_acc", factor=0.33, patience=2, verbose=1)
    callbacks_list = [checkpoint, reduce_lr]

    # Create generator from the dataframe
    train_generator = datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="class",
        target_size=(image_size, image_size),
        class_mode="categorical",
        batch_size=batch_size
    )

    valid_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filename",
        y_col="class",
        target_size=(image_size, image_size),
        class_mode="categorical",
        batch_size=batch_size
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    # Sanity check to make sure the class is in the right order 001,002,003..
    print(train_generator.class_indices)

    # Using fit_generator to avoid fitting all training data into memory
    # Start training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        shuffle=True,
        callbacks=callbacks_list,
        epochs=epochs,
    )

if __name__ == '__main__':
    main()