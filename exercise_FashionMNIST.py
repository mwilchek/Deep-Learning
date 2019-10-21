# %% -------------------------------------------------------------------------------------------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential,  load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# % -------------------------------------------------------------------------------------
# Fit a MLP to the FashionMNIST dataset: https://github.com/zalandoresearch/fashion-mnist
# % -------------------------------------------------------------------------------------
# Sets random seeds and some other stuff for reproducibility. Note even this might not give fully reproducible results.
# There seems to be a problem with the TF backend. However, the results should be very similar.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)


LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS = 20
BATCH_SIZE = 512
DROPOUT = 0.2


# 1. Download the data using keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
y_train, y_test = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 2. Try using more/less layers and different hidden sizes to get a good better fit. Also play with the dropout.
# Try different batch sizes and get a feeling of how they can influence convergence and speed

# Build model
model = Sequential([  # The dropout is placed right after the outputs of the hidden layers.
    Dense(N_NEURONS[0], input_dim=784, kernel_initializer=weight_init),  # This sets some of these outputs to 0, so that
    Activation("relu"),  # a random dropout % of the hidden neurons are not used during each training step,
    Dropout(DROPOUT),  # nor are they updated. The Batch Normalization normalizes the outputs from the hidden
    BatchNormalization()  # activation functions. This helps with neuron imbalance and can speed training significantly.
])  # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing.
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
# Adds a final output layer with softmax to map to the 10 classes
model.add(Dense(10, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])



# 3. Add an option to save the model on each epoch, and stop saving it when the validation
# loss begins to increase (early stopping) - https://keras.io/callbacks/: ModelCheckpoint

# --- Early stopping ---
early_stopping_monitor = EarlyStopping(patience=2)
results = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=N_EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping_monitor])


# 4. Add an option to only test the model, by loading the model you saved on the training phase
#--- Setting up directory --
base_dir = os.getcwd()

# Save model in HDF5 fomrat
model.save(os.path.join(base_dir, 'my_model.h5'))

# Deletes existing model
del model

# --- Load model ---
# Returns a compiled model identical to the previous one
model = load_model(os.path.join(base_dir, 'my_model.h5'))

# --- Saving only a models architecture ---
# Convert model to JSON
json_string = model.to_json()

# --- Reconstructing model from different formats ---
# JSON model reload
model_json = model_from_json(json_string)


# --- Saving/Loading Model's Weights (only) ---
# Vanilla save/load
model.save_weights(os.path.join(base_dir, 'my_model_weights.h5'))
model.load_weights(os.path.join(base_dir, 'my_model_weights.h5'))

# --- Transfer learning approach to re-load ---
# When we load weights into a different
# architeture with some common layers
model.load_weights(os.path.join(base_dir, 'my_model_weights.h5'),
                   by_name=True)

# --- Utility functions ---
print(model.output_shape)
print(model.get_config())
model.get_weights()
model.layers
model.layers[0].input_shape
model.layers[0].output_shape


# 5. Print out the confusion matrix
test_predictions = model.predict(x_test)
print("Confusion Matrix: \n", confusion_matrix(y_test.argmax(axis=1), test_predictions.argmax(axis=1)))

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")


# 6. Define a function to show some images that were incorrectly classified
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(['train', 'test'], loc='upper left')
plt.show()
