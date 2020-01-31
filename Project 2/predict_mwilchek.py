# %% --------------------------------------- Imports -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.system("sudo pip install --upgrade cv2")
os.system("sudo pip install --upgrade tqdm")
os.system("sudo pip install --upgrade pytorch")
import cv2
from tqdm import tqdm


def predict(x_test):
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    nrows = 64
    ncolumns = 64
    channels = 3

    processed_pics = []

    print("Reading in scoring images and converting to arrays...")
    for picture in tqdm(x_test):
        picture_path = os.path.abspath(picture)
        processed_pics.append(cv2.resize(cv2.imread(picture_path, cv2.IMREAD_COLOR), (nrows, ncolumns),
                                         interpolation=cv2.INTER_AREA))

    scoring_data = np.array(processed_pics)
    scoring_data = scoring_data.reshape(len(scoring_data), channels, nrows, ncolumns)
    scoring_data = torch.FloatTensor(scoring_data)
    print("Shape of images is: ", scoring_data.shape)

    # %% --------------------------------------------- Load Class ------------------------------------------------------
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            # 3 input image channel, 18 output channels, 3x3 square convolution
            # kernel
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
            self.conv3 = nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1))
            self.conv4 = nn.Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1))
            self.conv5 = nn.Conv2d(128, 160, kernel_size=(3, 3), stride=(1, 1))
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(160 * 5 * 5, 100)  # 5*5 from image dimension
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 7)

        def forward(self, x):
            # Max pooling over a (2, 2) window
            x = F.relu(self.conv1(x))
            # If the size is a square you can only specify a single number
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    # %% --------------------------------------------- Predict ---------------------------------------------------------
    print("Loading model that will be the highest score in the class...")
    model = Net()
    model.load_state_dict(torch.load('model_mwilchek.pt'))
    model.eval()
    m = nn.Sigmoid()
    y_pred = m(model(scoring_data))
    return y_pred
