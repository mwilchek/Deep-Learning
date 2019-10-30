import cv2
import os
import gc
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import rescale


# Resource used to help: https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).

pictures = 'Exam1/train/pics'
labels = 'Exam1/train/labels'

# Define pic parameters
nrows = 150  # 150 pixels by
ncolumns = 150  # 150 pixels
channels = 3  # using RGB colors

processed_pics = []
processed_random_flip_up_down = []
processed_random_saturate = []
processed_random_left_right = []
processed_random_brightness = []
processed_rotate_flip = []
processed_random_contrast = []
processed_random_hue = []
processed_random_all_colors = []
processed_scale_out = []
processed_noise = []

processed_labels = []
processed_labels_2 = []
processed_labels_3 = []
processed_labels_4 = []
processed_labels_5 = []
processed_labels_6 = []
processed_labels_7 = []
processed_labels_8 = []
processed_labels_9 = []
processed_labels_10 = []
processed_labels_11 = []

# Process Cell Pictures
picture_list = os.listdir(pictures)
picture_list.sort()

for cell_file in picture_list:
    path = os.path.join(pictures, cell_file)
    path = os.path.abspath(path)

    # Process original image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
    processed_pics.append(img)

    # Get new image and random flip up or down original
    rand_flip_down_up = tf.image.random_flip_up_down(img)
    rand_flip_down_up = np.array(rand_flip_down_up)
    processed_random_flip_up_down.append(rand_flip_down_up)

    # Get new image and random saturate the original
    rand_saturate = tf.image.random_saturation(img, 0.6, 1.6)
    rand_saturate = np.array(rand_saturate)
    processed_random_saturate.append(rand_saturate)

    # Get new image and random flip left or right original
    rand_flip_left_right = tf.image.random_flip_left_right(img)
    rand_flip_left_right = np.array(rand_flip_left_right)
    processed_random_left_right.append(rand_flip_left_right)

    # Get new image and random brighten original
    rand_brighten = tf.image.random_brightness(img, 0.05)
    rand_brighten = np.array(rand_brighten)
    processed_random_brightness.append(rand_brighten)

    # Get new image and random rotate/flip original
    rand_rotate = tf.image.rot90(img, k=1)
    rand_rotate = np.array(rand_rotate)
    processed_rotate_flip.append(rand_rotate)

    # Get new image and random contrast original
    rand_contrast = tf.image.random_contrast(img, 0.7, 1.3)
    rand_contrast = np.array(rand_contrast)
    processed_random_contrast.append(rand_contrast)

    # Get new image and random hue original
    rand_hue = tf.image.random_hue(img, 0.08)
    rand_hue = np.array(rand_hue)
    processed_random_hue.append(rand_hue)

    # Get new image and random color affect original
    rand_colors = tf.image.random_saturation(img, 0.6, 1.6)
    rand_colors = tf.image.random_brightness(rand_colors, 0.05)
    rand_colors = tf.image.random_contrast(rand_colors, 0.7, 1.3)
    rand_colors = tf.image.random_hue(rand_colors, 0.08)
    rand_colors = tf.image.rot90(rand_colors, k=2)

    rand_colors = np.array(rand_colors)
    processed_random_all_colors.append(rand_colors)

    # Get new image and scale out from original
    scale_out_pic = rescale(img, scale=2.0, mode='constant')
    crop_size = [nrows, ncolumns, channels]
    scale_out_pic = tf.image.random_crop(scale_out_pic, size=crop_size)
    scale_out_pic = np.array(scale_out_pic)
    processed_scale_out.append(scale_out_pic)

    # Get new image and add noise from original
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0,
                             dtype=tf.float32)
    noise_img = tf.add(img, noise)
    noise_img = np.array(noise_img)
    processed_noise.append(noise_img)

# Join all array images into master array
pics = processed_pics + processed_random_flip_up_down + processed_random_saturate + processed_random_left_right + \
       processed_random_brightness + processed_rotate_flip + processed_random_contrast + processed_random_hue + \
       processed_random_all_colors + processed_scale_out + processed_noise

# Process Labels
labels_list = os.listdir(labels)
labels_list.sort()

for text_file in labels_list:
    path = os.path.join(labels, text_file)

    cell_text = open(path)
    cell_label_string = cell_text.readline()

    label = "ERROR"

    # Assign label value per content
    if cell_label_string == 'red blood cell':
        label = 0
    if cell_label_string == 'ring':
        label = 1
    if cell_label_string == 'schizont':
        label = 2
    if cell_label_string == 'trophozoite':
        label = 3

    # If label value has not been changed from error raise error, else append to array
    if label != "ERROR":
        processed_labels.append(label)
        processed_labels_2.append(label)
        processed_labels_3.append(label)
        processed_labels_4.append(label)
        processed_labels_5.append(label)
        processed_labels_6.append(label)
        processed_labels_7.append(label)
        processed_labels_8.append(label)
        processed_labels_9.append(label)
        processed_labels_10.append(label)
        processed_labels_11.append(label)
    else:
        raise ValueError("Error with Label file: " + path)

# Join respective labels for all images to master list
labels = processed_labels + processed_labels_2 + processed_labels_3 + processed_labels_4 + processed_labels_5 + \
         processed_labels_6 + processed_labels_7 + processed_labels_8 + processed_labels_9 + processed_labels_10 + \
         processed_labels_11

X = np.array(pics)

y = np.array(labels)

print("Shape of images is: ", X.shape)
print("Shape of labels is: ", y.shape)

SEED = 666
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=0.2, stratify=y)

print("Shape of train images is: ", x_train.shape)
print("Shape of test images is: ", x_test.shape)
print("Shape of train labels is: ", y_train.shape)
print("Shape of test labels is: ", y_test.shape)

del X
del y
gc.collect()

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
