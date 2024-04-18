import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Dropout
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import VGG16

# Defining categories for labelling data
Categories = ['angry', 'happy', 'neutral', 'sad']

# Defining zoom factors
zoom_factors = [1.2, 1.4, 1.6, 2]

# Defining directory path for training data
datadir = 'C:/Users/maxvr/PycharmProjects/pythonProject2/NewDataSet'

def dynamic_preprocess(img, label):
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float64)
    img += noise
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)

    return img, label


def load_and_preprocess_image(path):
    """Loading image, resizing it to a more workable resolution, rotating it with a random angle between -20 and 20,
    and normalizing the image"""
    img = cv2.imread(path)

    height, width, channels = img.shape
    resize_height = int(height/5)
    resize_width = int(width/5)
    img = cv2.resize(img, (resize_width, resize_height))

    angle = random.randint(-20, 20)
    center = (resize_height//2, resize_width//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, rotation_matrix, (resize_width, resize_height))

    img = img / 255.0  # Normalize to [0, 1]

    return img


def zoom_in(image, zoom_factor):
    """Take an image and zoom in by zoom-factor on a section in the middle of the image"""
    # Get image dimensions
    height, width, channels = image.shape
    # Calculate new dimensions after zooming
    new_h = int(height / zoom_factor)
    new_w = int(width / zoom_factor)

    # Calculate the coordinates for cropping the middle portion
    top = (height - new_h) // 2
    bottom = top + new_h
    left = (width - new_w) // 2
    right = left + new_w

    # Crop the middle portion of the image
    cropped_image = image[top:bottom, left:right]

    # Resize the cropped image back to the original dimensions
    zoomed_image = cv2.resize(cropped_image, (width, height))

    return zoomed_image


def load_dataset(datadir, Categories):
    """Load the entire dataset, and augment data (rotate, zoom, and normalize)"""
    image_arr = []  # For storing all the images
    target_arr = []  # For storing labels

    # Looping through categories of images
    for i, category in enumerate(Categories):
        print(f'Loading... category: {category}')
        path = os.path.join(datadir, category)

        # Looping through all images in category
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            for f in range(8):
                img = load_and_preprocess_image(img_path)
                image_arr.append(img)
                target_arr.append(i)  # Using index of category as label
                for g in zoom_factors:
                    zoomed_img = zoom_in(img, g)
                    image_arr.append(zoomed_img)
                    target_arr.append(i)

        print(f'Loaded category: {category} successfully')

    # Convert lists to numpy arrays
    image_arr = np.array(image_arr)
    target_arr = np.array(target_arr)

    # One-hot encode the labels
    num_classes = len(Categories)
    target_arr = to_categorical(target_arr, num_classes=num_classes)

    # Splitting dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(image_arr, target_arr, test_size=0.20, random_state=24,
                                                        stratify=target_arr, shuffle=True)

    # Creating TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(96)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(96)

    return train_ds, test_ds

def model_training():
    """Initializing the CNN model, freezing VGG16 layers, and training the added layers"""
    train_ds, test_ds = load_dataset(datadir, Categories)
    train_ds = train_ds.map(dynamic_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(dynamic_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    input_shape = (96, 128, 3)

    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained VGG16 layers
    for layer in vgg16_model.layers:
        layer.trainable = False

    vgg16_output = vgg16_model.output

    flatten = Flatten()(vgg16_output)
    dense_1 = Dense(320, activation='relu', kernel_regularizer=l2(0.01))(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    output = Dense(4, activation="softmax")(drop_1)

    # Model compile
    model = Model(inputs=vgg16_model.input, outputs=output)
    model.compile(optimizer='adam', loss=["categorical_crossentropy"], metrics=['accuracy'])
    model.summary()

    # Configure ModelCheckpoint
    fle_s = 'C:/Users/maxvr/PycharmProjects/pythonProject2/archive (1)/Output/emotion_model105.keras'
    checkpointer = ModelCheckpoint(fle_s, monitor='loss', verbose=1, save_best_only=True,
                                   save_weights_only=False, mode='auto', save_freq='epoch')
    callback_list = [checkpointer]

    save = model.fit(train_ds, batch_size=96, validation_data=test_ds, epochs=15,
                     callbacks=[callback_list])

    # Save the entire model
    model.save('C:/Users/maxvr/PycharmProjects/pythonProject2/archive (1)/Output/emotion_model105.h5')

    train_loss = save.history['loss']
    test_loss = save.history['val_loss']
    train_accuracy = save.history['accuracy']
    test_accuracy = save.history['val_accuracy']

    # Plotting a line chart to visualize the loss and accuracy values by epochs.
    fig, ax = plt.subplots(ncols=2, figsize=(15, 7))
    ax = ax.ravel()
    ax[0].plot(train_loss, label='Train Loss', color='royalblue', marker='o', markersize=5)
    ax[0].plot(test_loss, label='Test Loss', color='orangered', marker='o', markersize=5)
    ax[0].set_xlabel('Epochs', fontsize=14)
    ax[0].set_ylabel('Categorical Crossentropy', fontsize=14)
    ax[0].legend(fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].plot(train_accuracy, label='Train Accuracy', color='royalblue', marker='o', markersize=5)
    ax[1].plot(test_accuracy, label='Test Accuracy', color='orangered', marker='o', markersize=5)
    ax[1].set_xlabel('Epochs', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[1].legend(fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    fig.suptitle(x=0.5, y=0.92, t="Lineplots showing loss and accuracy of CNN model by epochs", fontsize=16)
    plt.show()


model_training()
