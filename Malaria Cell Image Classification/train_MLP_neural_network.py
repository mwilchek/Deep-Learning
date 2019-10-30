# %% --------------------------------------- Imports -------------------------------------------------------------------
import os
import random
import talos
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from talos.model.normalizers import lr_normalizer
from talos.model.early_stopper import early_stopper
from talos import Evaluate

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 666
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# Resource used: https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb#six
# https://autonomio.github.io/docs_talos/#probabilistic-reduction
# https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
p = {'lr': (0.1, 10, 10),
     'first_neuron': [32, 64, 128],
     'batch_size': [512, 1024],
     'epochs': [30],
     'dropout': (0, 0.40, 10),
     'kernel_initializer': ['uniform', 'normal'],
     'weight_regulizer': [None],
     'emb_output_dims': [None],
     'optimizer': [Adam, Nadam, RMSprop, SGD],
     'loss': ['categorical_crossentropy'],
     'activation': ['relu', 'elu'],
     'last_activation': ['softmax']}

# %% -------------------------------------- Data Prep ------------------------------------------------------------------
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 150 * 150 * 3
x_train = x_train.reshape(-1, 67500)
x_test = x_test.reshape(-1, 67500)

y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)


# %% -------------------------------------- MLP Tuning ----------------------------------------------------------

def malaria_model(x_train, y_train, x_test, y_test, params):
    model = Sequential([
        Dense(params['first_neuron'], input_shape=(x_train.shape[1],), kernel_initializer=params['kernel_initializer']),
        Activation(params['activation']),
        Dropout(params['dropout']),
        BatchNormalization()
    ])

    N_NEURONS = (200, 100)

    # # Loops over the hidden dims to add more layers
    for n_neurons in N_NEURONS[0:]:
        model.add(Dense(n_neurons, activation=params['activation'], kernel_initializer=params['kernel_initializer']))
        model.add(Dropout(params['dropout'], seed=SEED))
        model.add(BatchNormalization())

    # Adds a final output layer with softmax to map to the 4 classes
    model.add(Dense(y_train.shape[1], activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))

    # model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                  loss=params['loss'], metrics=['acc'])  # 'accuracy'
    # %% -------------------------------------- Training Loop ----------------------------------------------------------
    history = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'],
                        validation_data=(x_test, y_test),
                        callbacks=[ModelCheckpoint("mlp_mwilchek.hdf5", monitor="val_loss", save_best_only=True),
                                   early_stopper(params['epochs'], mode='moderate')])

    return history, model


# %% ------------------------------------------ MLP Tuning Eval --------------------------------------------------------

# and run the experiment
t = talos.Scan(x=x_train,
               y=y_train,
               x_val=x_test,
               y_val=y_test,
               model=malaria_model,
               params=p,
               experiment_name='malaria_cells_classification',
               # round_limit=10  # just does 10 rounds of modeling / 10 different param configs
               fraction_limit=.05)  # just does 5% of total number param configs)

results = talos.Evaluate(t)
results_df = results.data
results_df = results_df.sort_values(by='val_acc', ascending=False)
results_df.to_csv('tuning_results.csv')

# %% -------------------------------------- Final Model  ----------------------------------------------------------
# After reviewing tuning results from the above, define best model parameters
final_model = Sequential([
    Dense(64, input_shape=(x_train.shape[1],), kernel_initializer='uniform'),
    Activation('relu'),
    Dropout(0.00),
    BatchNormalization()
])

# Define hidden layers
N_NEURONS = (200, 100)
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[0:]:
    final_model.add(Dense(n_neurons, activation='relu', kernel_initializer='uniform'))
    final_model.add(Dropout(0.00, seed=SEED))
    final_model.add(BatchNormalization())

# Adds a final output layer with softmax to map to the 4 classes
final_model.add(Dense(y_train.shape[1], activation='softmax',
                      kernel_initializer='uniform'))

final_model.compile(optimizer=Nadam(lr=lr_normalizer(1.0900, Nadam)), loss='categorical_crossentropy', metrics=['acc'])
# %% -------------------------------------- Training Loop ----------------------------------------------------------
final_model.fit(x_train, y_train, batch_size=512, epochs=30,
                validation_data=(x_test, y_test),
                callbacks=[ModelCheckpoint("mlp_mwilchek.hdf5", monitor="val_loss", save_best_only=True),
                           early_stopper(30, mode='moderate')])

print("Final accuracy on validations set:", 100 * final_model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(final_model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
print("F1 score", f1_score(np.argmax(final_model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))

# --- Setting up directory --
base_dir = os.getcwd()

# Save model in HDF5 fomrat
final_model.save(os.path.join(base_dir, 'mlp_mwilchek.hdf5'))
