# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation

# -------------------------------------
# Approximate a 3D Function using a MLP
# -------------------------------------

LR = 0.1
N_NEURONS = 10
N_EPOCHS = 1000


# 1. Define the function y = x1**2 - x2**2 you will train the MLP to approximate. A 3D plot can be found at:
# # http://www.livephysics.com/tools/mathematical-tools/online-3-d-function-grapher/

x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-2, 2, 100)
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

    ax.view_init(45, 280)
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
X = np.vstack((X1.flatten(), X2.flatten())).T.reshape(10000, 2)
y = Y.reshape(10000, 1)


# 4. Define a MLP to approximate this function using the data you just generated.
# Play with the number of layers, neurons and hidden activation functions (tanh, ReLu, etc.)
model = Sequential([
    Dense(N_NEURONS, input_dim=2),  # Linear layer that maps 1 input dim to N_NEURONS hidden dim
    Activation("relu"),  # Sigmoid hidden transfer function
    Dense(1)  # Maps N_NEURONS hidden dim to 1 output dim
])
model.summary()

# 5. Use Adam or another optimizer and train the network. Find an appropriate learning rate and number of epochs.
model.compile(optimizer=Adam(lr=LR),
              loss='mean_squared_error')

# Train neural network
history_train = model.fit(X, y, batch_size=10000, epochs=N_EPOCHS)

# Get loss function
loss = history_train.history['loss']

# Setup grid for plotting epochs
epochs = range(1, len(loss) + 1)

# Plot of epochs
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss and Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

# Visually shows the approximation obtained by the MLP
plt.title("MLP fit to $y = x_1^2 - x_2^2$ | MSE: {:.5f}".format(history_train.history["loss"][-1]))
plt.xlabel("x")
plt.ylabel("y")
plt.plot(X, y, label="Real Function")
plt.plot(X, model.predict(X), linestyle="dashed", label="MLP Approximation")
plt.legend()
plt.show()
plt.close()


# 6. Use the function you defined in 2. to visualize how well your MLP fits the original function
helper_function(X1, X2, model.predict(X).reshape(100, 100))
