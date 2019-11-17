# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import torch
import torch.nn as nn

# -------------------------------------
# Approximate a 3D Function using a MLP
# -------------------------------------
LR = 2.5e-1
N_NEURONS = 10
N_EPOCHS = 50000
PRINT_LOSS_EVERY = 1000

# %% -------------------------------------- MLP Class ------------------------------------------------------------------
# Defines model class. Inherits from nn.Module to get all useful methods and make the class compatible with PyTorch
class MLP(nn.Module):
    """ MLP with 2 layers, sigmoid transfer function and hidden_dim neurons on the hidden layer"""
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()  # Initializes nn.Module
        # Linear Layer that maps input (n_examples, 1) to hidden_dim (number of neurons)
        self.linear1 = nn.Linear(2, N_NEURONS)
        self.act1 = torch.sigmoid  # Hidden transfer function
        # Linear Layer that maps hidden_dim to output (n_examples, 1), i.e, 1 neuron
        self.linear2 = nn.Linear(N_NEURONS, 1)

    # Calling this method "forward" will make it possible to go forward by just calling model(x) (see below)
    def forward(self, x):
        # Just a sequential pass to go through the 2 layers
        return self.linear2(self.act1(self.linear1(x)))


# 1. Define the function y = x1**2 - x2**2 you will train the MLP to approximate. A 3D plot can be found at:
# # http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/

# Gets the input interval and the targets
x1 = np.linspace(-1, 1, 100)
x2 = np.linspace(-1, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = X1 ** 2 - X2 ** 2

# 2. Define a helper function to plot the real function and the MLP approximation. Hint:
# from mpl_toolkits.mplot3d import Axes3D, use ax.contour3D on 3 inputs with shapes (sqrt(n_examples), sqrt(n_examples))
# You may do 3. first to get the data and figure out why the shapes are like this
def helper_function(X1, X2, Y):
    fig = plt.figure(figsize=(16, 8))

    # Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X1, X2, Y, rstride=5, cstride=5, cmap='jet', alpha=.4, edgecolor='none')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title("Surface Plot")

    # Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X1, X2, Y, 50, cmap='jet')
    ax.set_title("Contour Plot")

    # Plotting the iterations and intermediate values
    plt.show()
    plt.close()

helper_function(X1, X2, Y)

# 3. Generate the data to train the network using the function you defined in 1. Hint:
# Use np.meshgrid() and then reshape the input to (n_examples, 2) and the target to (n_examples, 1)
X1, X2 = np.meshgrid(x1, x2)
X1 = torch.Tensor(X1).reshape(10000, 1)
X2 = torch.Tensor(X2).reshape(10000, 1)
X = torch.stack([X1, X2], dim=2).T.reshape(10000, 2)
X.requires_grad = True
y = torch.Tensor(Y).reshape(10000, 1)
# 4. Define a MLP to approximate this function using the data you just generated.
# Initializes model and moves it to GPU if available
model = MLP(N_NEURONS)
# Initializes a Gradient Descent Optimizer with default hyper-parameters
# We pass the model.parameters() generator so that all the model's parameters are updated
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Initializes a Mean Square Error performance index
criterion = nn.MSELoss()

# Starts the training loop
for epoch in range(N_EPOCHS):
    # Sets the gradients stored on the .grad attribute of each parameter from the previous iteration to 0
    optimizer.zero_grad()  # It is good practice to do it right before going forward on any model
    # Goes forward (doing full batch here), notice we don't need to do model.forward(p)
    t_pred = model(X)
    # Computes the mse
    loss = criterion(y, t_pred)
    # Goes backwards (computes all the gradients of the mse w.r.t the parameters
    # starting from the output layer all the way to the input layer)
    loss.backward()
    # Updates all the parameters using the gradients which were just computed
    optimizer.step()
    # Checks the training process
    if epoch % PRINT_LOSS_EVERY == 0:
        print("Epoch {} | Loss {:.5f}".format(epoch, loss.item()))


# 5. Use Adam or another optimizer and train the network. Find an appropriate learning rate and number of epochs.
# Visually shows the approximation obtained by the MLP
plt.title("MLP fit to $y = x_1^2 - x_2^2$ | MSE: {:.5f}".format(loss.item()))
plt.xlabel("x")
plt.ylabel("y")
# .detach() to take the Tensors out of the computational graph
# .numpy() to convert the Tensors to NumPy arrays
plt.plot(X.detach().numpy(), y.numpy(), label="Real Function")
plt.plot(X.detach().numpy(), t_pred.detach().numpy(), linestyle="dashed", label="MLP Approximation")
plt.legend()
plt.show()

# 6. Use the function you defined in 2. to visualize how well your MLP fits the original function
helper_function(X1.detach().numpy().reshape(100, 100), X2.detach().numpy().reshape(100, 100),
                t_pred.detach().numpy().reshape(100, 100))

# %% -------------------------------------------------------------------------------------------------------------------


