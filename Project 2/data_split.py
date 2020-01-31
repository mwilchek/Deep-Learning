import cv2
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
from tqdm import tqdm

data_list = pd.read_csv('/home/ubuntu/Desktop-Sync-Folder/Exam2/train_data_with_labels.csv')

# Define pic parameters
nrows = 64  # 32 pixels by
ncolumns = 64  # 32 pixels
channels = 3  # using RGB colors
augmentation_rounds = 20

def process_originals(data_list, nrows, ncolumns):
    picture_list = data_list['Picture_Path']

    processed_pics = []
    processed_labels = np.array(data_list.drop(['Picture_Path', 'Labels'], axis=1))

    print("Reading in original images and processing non-randomized augmentation...")
    for pic in tqdm(picture_list):
        # Process original image
        img = cv2.imread(pic, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_CUBIC)
        processed_pics.append(img)

    # Join all array images into master array
    pics = processed_pics

    # X = np.array(processed_pics)
    X = np.array(pics)

    y = np.array(processed_labels)

    return X, y

def process_originals(data_list, nrows, ncolumns):
    picture_list = data_list['Picture_Path']

    processed_pics = []
    processed_random_flip_up_down = []
    processed_random_left_right = []
    processed_rotate_flip = []

    processed_labels = np.array(data_list.drop(['Picture_Path', 'Labels'], axis=1))
    processed_labels_2 = processed_labels
    processed_labels_3 = processed_labels
    processed_labels_4 = processed_labels

    print("Reading in original images and processing non-randomized augmentation...")
    for pic in tqdm(picture_list):
        # Process original image
        img = cv2.imread(pic, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_AREA)
        processed_pics.append(img)

        # Get new image and random flip up or down original
        rand_flip_down_up = tf.image.random_flip_up_down(img)
        rand_flip_down_up = np.array(rand_flip_down_up)
        processed_random_flip_up_down.append(rand_flip_down_up)

        # Get new image and random flip left or right original
        rand_flip_left_right = tf.image.random_flip_left_right(img)
        rand_flip_left_right = np.array(rand_flip_left_right)
        processed_random_left_right.append(rand_flip_left_right)

        # Get new image and random rotate/flip original
        rand_rotate = tf.image.rot90(img, k=1)
        rand_rotate = np.array(rand_rotate)
        processed_rotate_flip.append(rand_rotate)

    # Join all array images into master array
    pics = processed_pics + processed_random_flip_up_down + processed_random_left_right + processed_rotate_flip

    # X = np.array(processed_pics)
    X = np.array(pics)

    y = np.array(processed_labels)
    y = np.append(y, processed_labels_2, axis=0)
    y = np.append(y, processed_labels_3, axis=0)
    y = np.append(y, processed_labels_4, axis=0)

    return X, y


def Augment_Images(data_list, nrows, ncolumns, channels):
    picture_list = data_list['Picture_Path']

    processed_random_saturate = []
    processed_random_brightness = []
    processed_random_contrast = []
    processed_random_all_colors = []
    processed_scale_out = []
    processed_noise = []
    processed_all = []
    processed_contrast_scale_out = []

    processed_labels = np.array(data_list.drop(['Picture_Path', 'Labels'], axis=1))
    processed_labels_2 = processed_labels
    processed_labels_3 = processed_labels
    processed_labels_4 = processed_labels
    processed_labels_5 = processed_labels
    processed_labels_6 = processed_labels
    processed_labels_7 = processed_labels
    processed_labels_8 = processed_labels
    processed_labels_9 = processed_labels

    print("Reading in original images and processing randomized augmentation...")
    for pic in tqdm(picture_list):
        # Process original image
        img = cv2.imread(pic, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (nrows, ncolumns), interpolation=cv2.INTER_AREA)

        # Get new image and random saturate the original
        rand_saturate = tf.image.random_saturation(img, 0.4, 1.8)
        rand_saturate = np.array(rand_saturate)
        processed_random_saturate.append(rand_saturate)

        # Get new image and random brighten original
        rand_brighten = tf.image.random_brightness(img, 0.05)
        rand_brighten = np.array(rand_brighten)
        processed_random_brightness.append(rand_brighten)

        # Get new image and random contrast original
        rand_contrast = tf.image.random_contrast(img, 0.4, 1.8)
        rand_contrast = np.array(rand_contrast)
        processed_random_contrast.append(rand_contrast)

        # Get new image and random color affect original
        rand_colors = tf.image.random_saturation(img, 0.4, 1.8)
        rand_colors = tf.image.random_brightness(rand_colors, 0.05)
        rand_colors = tf.image.random_contrast(rand_colors, 0.4, 1.8)
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

        # Get noise image and apply color changes
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0,
                                 dtype=tf.float32)
        noise_img = tf.add(img, noise)
        rand_colors = tf.image.random_saturation(noise_img, 0.4, 1.8)
        rand_colors = tf.image.random_brightness(rand_colors, 0.05)
        rand_colors = tf.image.random_contrast(rand_colors, 0.4, 1.8)

        rand_colors = np.array(rand_colors)
        processed_all.append(rand_colors)

        # Get new image and random contrast original, then scale
        rand_contrast = tf.image.random_contrast(img, 0.8, 1.3)
        rand_contrast = np.array(rand_contrast)
        scale_out_pic = rescale(rand_contrast, scale=2.0, mode='constant')
        crop_size = [nrows, ncolumns, channels]
        scale_out_pic = tf.image.random_crop(scale_out_pic, size=crop_size)
        scale_out_pic = np.array(scale_out_pic)
        processed_contrast_scale_out.append(scale_out_pic)

    # Join all array images into master array
    pics = processed_random_saturate + processed_random_brightness + processed_random_contrast + \
           processed_random_all_colors + processed_scale_out + processed_noise + processed_all + \
           processed_contrast_scale_out

    # X = np.array(processed_pics)
    X = np.array(pics)

    # y = np.array(processed_labels)
    y = np.append(processed_labels_2, processed_labels_3, axis=0)
    y = np.append(y, processed_labels_4, axis=0)
    y = np.append(y, processed_labels_5, axis=0)
    y = np.append(y, processed_labels_6, axis=0)
    y = np.append(y, processed_labels_7, axis=0)
    y = np.append(y, processed_labels_8, axis=0)
    y = np.append(y, processed_labels_9, axis=0)

    return X, y


# Process original images with augmentation that cannot be randomized
x, y = process_originals(data_list, nrows, ncolumns)
x_master = x
y_master = y

del x, y
gc.collect()

# Randomize Data Augmentation, combine to master, and do it 33 times
for i in range(1, augmentation_rounds):
    print("Augmenting data round " + str(i) + " of " + str(augmentation_rounds))
    temp_x, temp_y = Augment_Images(data_list, nrows, ncolumns, channels)
    x_master = np.append(x_master, temp_x, axis=0)
    y_master = np.append(y_master, temp_y, axis=0)

    del temp_x, temp_y
    gc.collect()

print("")
print("Shape of final image dataset is: ", x_master.shape)
print("Shape of final labels for dataset is: ", y_master.shape)
print("")

SEED = 666
x_train, x_test, y_train, y_test = train_test_split(x_master, y_master, random_state=SEED, test_size=0.2, shuffle=True)

print("Shape of train images is: ", x_train.shape)
print("Shape of test images is: ", x_test.shape)
print("Shape of train labels is: ", y_train.shape)
print("Shape of test labels is: ", y_test.shape)

del x_master
del y_master
gc.collect()

np.save("/home/ubuntu/Desktop-Sync-Folder/Exam2/x_train.npy", x_train)
np.save("/home/ubuntu/Desktop-Sync-Folder/Exam2/y_train.npy", y_train)
np.save("/home/ubuntu/Desktop-Sync-Folder/Exam2/x_test.npy", x_test)
np.save("/home/ubuntu/Desktop-Sync-Folder/Exam2/y_test.npy", y_test)
