# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
from keras.models import load_model
import os

os.system("sudo pip install --upgrade cv2")
os.system("sudo pip install --upgrade talos")
import cv2
import talos
from talos.model.normalizers import lr_normalizer
from talos.model.early_stopper import early_stopper


def predict(x_test):
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    nrows = 150
    ncolumns = 150
    channels = 3

    processed_pics = []

    for picture in x_test:
        picture_path = os.path.abspath(picture)
        processed_pics.append(cv2.resize(cv2.imread(picture_path, cv2.IMREAD_COLOR), (nrows, ncolumns),
                                         interpolation=cv2.INTER_CUBIC))

    x = np.array(processed_pics)
    print("Shape of images is: ", x.shape)

    # Reshape features to appropriate dimension for model
    x = x.reshape(-1, 67500)

    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_mwilchek.hdf5')
    # If using more than one model to get y_pred, they need to be named as "mlp_ajafari1.hdf5", ""mlp_ajafari2.hdf5",
    # etc.
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model
    # If using more than one model to get y_pred, do the following:
    # return y_pred, model1, model2  # If you used two models
    # return y_pred, model1, model2, model3  # If you used three models, etc.

