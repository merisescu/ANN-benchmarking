# Hyperparameter optimisation for CNN to achieve minimum energy use

#%% Imports

# Setup

import os
os.environ["KERAS_BACKEND"] = "torch"

# Maths

import numpy as np
import time

#from sklearn.metrics import mean_squared_error as MSE
#from sklearn.metrics import mean_absolute_error as MAE
#from sklearn.metrics import explained_variance_score
#from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from calflops import calculate_flops # basic flops and macs estimation based on the model graph

# CNN
import keras
from keras import layers
import keras_tuner

# Visualisation

import matplotlib.pyplot as plt

#%% Script options
# enable to add verbose output and reproducible seeds
debug_mode = True
seed = 1

# CNN parameters
num_filters = 2 # hard coding this for now to match the effective number of filters present in N-MNIST

# Training parameters
batch_size = int(128)
#epochs = 5
optimizer = "adam"
loss_function = "categorical_crossentropy"

# Data evaluation parameters
evalute_training_set = False # Evaluates using the full training set every epoch without dropout and batch norm giving accurate training accuracy but very slow
#%% Extra function definitions

# Create the network structure
# TODO: make all values customisable
def build_CNN(hp): # define using hp.[dtype]("[hyperparameter]") to read in to keras tuner
    current_shape = input_shape[0] # hardcoded for square images for now
    model_params = [keras.Input(shape=input_shape)]
    pool_size = hp.Int("pool_size", min_value=1, max_value=4, step=1)
    kernel_size = hp.Int("kernel_size", min_value=3, max_value=5, step=2)
    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)
    for _ in range(num_layers):
        if current_shape > kernel_size:
            model_params.append(layers.Conv2D(num_filters, kernel_size=kernel_size, activation=hp.Choice("activation", values=["relu", "tanh", "sigmoid"])))
            current_shape -= kernel_size - 1
        if pool_size > 1 and current_shape >= 2*pool_size:
            model_params.append(layers.MaxPooling2D(pool_size=pool_size))
            current_shape /= pool_size
    model_params.append(layers.Flatten())
    model_params.append(layers.Dropout(0.5))
    model_params.append(layers.Dense(num_classes, activation="softmax"))
    model = keras.Sequential(model_params)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    #model.summary()
    return model

# Custom hypermodel to add metric to fit
# TODO: make all components customisable (early stop)
class CNN(keras_tuner.HyperModel):

    def build(self, hp):
        return build_CNN(hp)

    def fit(self, hp, model, *args, **kwargs):

        # Update callbacks kwarg without causing issues

        callbacks = kwargs.pop("callbacks", [])

        early_stop = keras.callbacks.EarlyStopping(
                                monitor="val_accuracy",
                                mode="max",
                                patience=3,
                                restore_best_weights=True,
                            )

        callbacks.append(early_stop)

        kwargs["callbacks"] = callbacks

        '''history = model.fit(
            *args,
            **kwargs
        )'''

        history = super().fit(hp,model,
                    *args,
                    **kwargs
                )

        _, macs, _ = calculate_flops(
                            model=model,
                            input_shape=(batch_size,) + input_shape,
                            output_as_string=False
                        )

        epochs = early_stop.best_epoch + 1

        history.history["train_efficiency"] = [macs*epochs] * len(history.history["loss"])
        history.history["infer_efficiency"] = [macs] * len(history.history["loss"])

        return history

#%% Load MNIST dataset
# From Keras tutorial  https://keras.io/examples/vision/mnist_convnet/

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1) # 28 px * 28 px * 1 channel (grayscale)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # tuple of np arrays 
# ((6000,28,28), (60000,)) , ((10000,28,28), (10000,))
# x: data, y: labels

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
if debug_mode:
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# split training data into training and validation sets, for compatibility with profiling (planned)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=debug_mode)

#%% Hyperparameter optimisation

# Define hyperparameter optimisation goals and constraints

tuner = keras_tuner.RandomSearch(
    hypermodel=CNN(),
    objective=[keras_tuner.Objective("train_efficiency","min"),
               keras_tuner.Objective("infer_efficiency","min"),
               keras_tuner.Objective("val_accuracy","max")
    ],
    max_trials=3, # number of hyperparameter combinations to try
    executions_per_trial=1, # number of times to train each model
    directory="CNN_hyperparams",
    project_name="experiment0",
    overwrite=True # very important!!! DO NOT TURN OFF OR BULLSHIT WILL HAPPEN
)

tuner.search(x_train, y_train, epochs = 6, validation_data=(x_val, y_val), batch_size=128)

#%% Evaluate model

models = tuner.get_best_models(num_models=2)
best_model = models[0]
print("-------------------------------------------------")
print("Best model summary")
best_model.summary()

# TODO: Verify optimisation is working correctly (actually getting lowest FLOPs/MACs with highest accuracy), related to below
# TODO: add data extraction from tuner.oracle

#%% Visualisation

