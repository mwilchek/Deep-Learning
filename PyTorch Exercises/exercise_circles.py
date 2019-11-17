# %% -------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

LR = 0.02
N_NEURONS = 3
N_EPOCHS = 10000
PRINT_LOSS_EVERY = 1000

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2, hidden_dim)
        self.act1 = torch.tanh
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.linear2(self.act1(self.linear1(x)))
# ---------------------------------------------
# Learn a circular Decision Boundary using  MLP
# ---------------------------------------------

# 1. Define a function to generate the y-points for a circle, taking as input the x-points and the radius r.
def get_single_circle(radius):
    angles = np.arange(0, 360, 1)
    x1 = np.cos(angles) * radius
    y1 = np.sin(angles) * radius

    return x1, y1

# 2. Use this function to generate the data to train the network. Label points with r=2 as 0 and points with r=4 as 1.
# Note that for each value on the x-axis there should be two values on the y-axis, and vice versa.
x1, y1 = get_single_circle(radius=2)
x2, y2 = get_single_circle(radius=4)
c1 = np.column_stack((x1, y1))
c2 = np.column_stack((x2, y2))
X = np.concatenate([c1, c2], axis=0)
y = np.zeros(X.shape[0])
y[c1.shape[0]:] = 1

# plot the training points
a, b = list(list(zip(*X)))
plt.scatter(a, b, c=y, cmap='RdBu')
plt.show()

X = torch.FloatTensor(X)
X.requires_grad = True
y = torch.Tensor(y).long()

# 3. Choose the right number of input and output neurons, define and train a MLP to classify this data.
model = MLP(N_NEURONS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# A classification problem is usually approached with a Categorical Cross-Entropy performance index
criterion = nn.CrossEntropyLoss()  # This one combines both the loss and the Log-SoftMax output function,
# which computes the probabilities of each example belonging to each class
# This is the preferred way as its computation is more stable. Thus, there is no need to include the
# SoftMax/Log-SoftMax output function on the model itself

for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))

# %% -------------------------------------- Check Approx ---------------------------------------------------------------
print(accuracy_score(y.numpy(), np.argmax(logits.detach().numpy(), axis=1))*100)

# 4. Make a contour plot of the MLP as a function of the x and y axis. You can follow
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html
x_span = np.linspace(-5, 5, 1000)
y_span = np.linspace(-5, 5, 1000)
xx, yy = np.meshgrid(x_span, y_span)
X_Y_span = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
logit = model(X_Y_span)
labels = np.argmax(logit.detach().numpy(), axis=1)
z = labels.reshape(xx.shape)
fig, ax = plt.subplots()
ax.contourf(xx, yy, z, cmap='RdBu', alpha=0.5)

# Get predicted labels on training data and plot
ax.scatter(a, b, c=y, cmap='RdBu', lw=0)
plt.show()
# %% -------------------------------------------------------------------------------------------------------------------