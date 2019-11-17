# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5


# % -------------------------------------------------------------------------------------
# Fit a CNN to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# % -------------------------------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100 * accuracy_score(y.cpu().numpy(), pred_labels)


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))  # output (n_examples, 16, 26, 26)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 13, 13)
        self.conv2 = nn.Conv2d(16, 32, (3, 3))  # output (n_examples, 32, 11, 11)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))  # output (n_examples, 32, 5, 5)
        self.linear1 = nn.Linear(32 * 5 * 5, 400)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 10)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)


# 1. Download the data using datasets.FashionMNIST
data_train = datasets.MNIST(root='.', train=True, download=True)
# Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
x_train, y_train = data_train.data.view(len(data_train), 1, 28, 28).float().to(device), data_train.targets.to(device)
x_train.requires_grad = True
data_test = datasets.MNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.view(len(data_test), 1, 28, 28).float().to(device), data_test.targets.to(device)

# 2. Run the same network as the one from the example.
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
        epoch, loss_train / BATCH_SIZE, acc(x_train, y_train), loss_test, acc(x_test, y_test)))

# 3. Try modifying the architecture and the hyper-parameters to get a better performance.
# Try including more conv layers and more kernels in each layer. This can allow for less MLP layers at the end.
# To do so, you will need to play around with zero-padding and maybe stack two conv layers together without any pooling.
# You can also remove the pooling and the MLP, and replace it with a final Global Average Pooling layer.


# %% -------------------------------------------------------------------------------------------------------------------
