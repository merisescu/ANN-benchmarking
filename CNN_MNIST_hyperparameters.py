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
batch_size = 128
#epochs = 5
optimizer = "adam"
loss_function = "categorical_crossentropy"

# Data evaluation parameters
evalute_training_set = False # Evaluates using the full training set every epoch without dropout and batch norm giving accurate training accuracy but very slow
#%% Extra function definitions

# Create the network structure
def build_CNN(hp): # define using hp.[dtype]("[hyperparameter]") to read in to keras tuner
    model_params = [keras.Input(shape=input_shape)]
    pool_size =hp.Int("pool_size", min_value=1, max_value=4, step=1)
    for _ in range(hp.Int("num_layers", min_value=1, max_value=8, step=1)):
        model_params.append(layers.Conv2D(num_filters, kernel_size=hp.Int("kernel_size", min_value=3, max_value=7, step=2), activation=hp.Choice("activation", values=["relu", "tanh", "sigmoid"])))
        if pool_size > 1:
            model_params.append(layers.MaxPooling2D(pool_size=pool_size))
    model_params.append(layers.Flatten())
    model_params.append(layers.Dropout(0.5))
    model_params.append(layers.Dense(num_classes, activation="softmax"))
    model = keras.Sequential(model_params)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    return model

# Custom callback to allow for a hyperparameter optimisation metric that is only called once per trial
class efficiencyCallback(keras.callbacks.Callback):

    def __init__(self):
        self.best_epoch = None
        self.best_value = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        value = logs.get("val_accuracy")

        if value is not None and value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        # output: flops, macs, params
        _, macs, _ = calculate_flops(
        model=self.model,
        input_shape=(batch_size,) + input_shape,
        output_as_string=True,
        )

        logs = logs or {}
        logs["train_efficiency"] = macs * self.best_epoch
        logs["infer_efficiency"] = macs

# Custom tuner to optimize hyperparameters for energy efficiency (FLOPs/MACs)
# using random search for now to develop
# currently using custom callback to implement this feature but as the tuner method is more flexible I'm keeping it around in case it becomes needed
'''
class efficiency(keras_tuner.randomSearch):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="efficiency", **kwargs)

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        score = calculate_flops(
                    model=model,
                    input_shape=(batch_size,) + input_shape,
                    output_as_string=True,
                )
        
        return self.hypermodel.fit(hp, model, *args, **kwargs)

    def result(self):
        return
'''

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

#%% Initialise network
# Adapted from Keras tutorial  https://keras.io/examples/vision/mnist_convnet/

model = build_CNN(keras_tuner.HyperParameters()) # test build

if debug_mode:
    model.summary()


#%% Hyperparameter optimisation

# Define hyperparameter optimisation goals and constraints
tuner = keras_tuner.RandomSearch(
    hypermodel=build_CNN,
    objective=[keras_tuner.Objective("train_efficiency","max"),
               keras_tuner.Objective("infer_efficiency","max"),
               keras_tuner.Objective("val_accuracy","max")
    ],
    max_trials=3, # number of hyperparameter combinations to try
    executions_per_trial=1, # number of times to train each model
)

tuner.search(x_train, y_train, epochs = 5, validation_data=(x_val, y_val),callbacks=efficiencyCallback())


#%% Evaluate model

models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()

#%% Visualisation

