# %% -------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.initializers import glorot_uniform

# ---------------------------------------------
# Learn a circular Decision Boundary using  MLP
# ---------------------------------------------

N_NEURONS = 10
LR = 0.2
N_EPOCHS = 1000


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

Y = to_categorical(y, num_classes=2)


# 3. Choose the right number of input and output neurons, define and train a MLP to classify this data.
model = Sequential([
    Dense(N_NEURONS, input_dim=2, kernel_initializer=glorot_uniform(42)),
    Activation("tanh"),
    Dense(2, kernel_initializer=glorot_uniform(42)),  # Output layer with softmax to map to the two classes
    Activation("softmax")
])

# Compiles using categorical cross-entropy performance index and tracks the accuracy during training
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X, Y, batch_size=len(X), epochs=N_EPOCHS)


# 4. Use model.evaluate to get the final accuracy on the whole set and print it out
score = model.evaluate(X, Y, batch_size=len(X))
predict_probabilities = model.predict(X, batch_size=len(X))
predict_classes = model.predict_classes(X, batch_size=len(X))
print("Accuracy is: " + str(score))


# 5. Make a contour plot of the MLP as a function of the x and y axis. You can follow
# https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html

x_span = np.linspace(-5, 5, 1000)
y_span = np.linspace(-5, 5, 1000)
xx, yy = np.meshgrid(x_span, y_span)
labels = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
z = labels.reshape(xx.shape)
fig, ax = plt.subplots()
ax.contourf(xx, yy, z, cmap='RdBu', alpha=0.5)

# Get predicted labels on training data and plot
ax.scatter(a, b, c=y, cmap='RdBu', lw=0)
plt.show()
