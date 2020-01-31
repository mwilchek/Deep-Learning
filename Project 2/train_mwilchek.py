# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import gc
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from keras.initializers import glorot_uniform
from tqdm import tqdm
from Exam2.pytorchtools import EarlyStopping
import torch.nn.functional as F

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 666
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train = np.load("Exam2/x_train.npy")
y_train = np.load("Exam2/y_train.npy")
x_test = np.load("Exam2/x_test.npy")
y_test = np.load("Exam2/y_test.npy")

# Define pic parameters
nrows = 64  # 32 pixels by
ncolumns = 64  # 32 pixels
channels = 3  # using RGB colors

x_train = x_train.reshape(len(x_train), channels, nrows, ncolumns)
x_test = x_test.reshape(len(x_test), channels, nrows, ncolumns)

x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()
x_train_tensor.requires_grad = True

del x_train
del x_test
del y_train
del y_test
gc.collect()


# %% -------------------------------------- Define Model  --------------------------------------------------------------


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
        x = F.elu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv5(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.celu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# %% -------------------------------------- Train Model  ---------------------------------------------------------------
def train_model(N_EPOCHS, model, BATCH_SIZE, patience, optimizer, criterion):
    m = nn.Sigmoid()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print("Starting training loop...")
    for epoch in range(N_EPOCHS):
        loss_train = 0
        model.train()
        for batch in tqdm(range(len(x_train_tensor) // BATCH_SIZE + 1)):
            inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
            optimizer.zero_grad()
            logits = m(model(x_train_tensor[inds]))
            loss = criterion(torch.sigmoid(logits), y_train_tensor[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_test_pred = m(model(x_test_tensor))
            loss = criterion(torch.sigmoid(y_test_pred), y_test_tensor)
            loss_test = loss.item()

        print("Epoch {} | Train Loss {:.5f} - Test Loss {:.5f}".format(
            epoch, loss_train / BATCH_SIZE, loss_test))

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(loss_test, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    # return trained model
    return model


# %% -------------------------------------- Parameters  ----------------------------------------------------------------
LR = 5e-3
N_EPOCHS = 6
BATCH_SIZE = 1024  # 1024
patience = 5
DROPOUT = 0.5
model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.BCELoss()

model = train_model(N_EPOCHS, model, BATCH_SIZE, patience, optimizer, criterion)

# %% -------------------------------------- Save Model  ---------------------------------------------------------------

# Save model in .pt fomrat
# torch.save(model, '/home/ubuntu/Desktop-Sync-Folder/Exam2/model_mwilchek.pt')
torch.save(model.state_dict(), '/home/ubuntu/Desktop-Sync-Folder/Exam2/model_mwilchek.pt')
