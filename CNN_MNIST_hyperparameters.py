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
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[efficiency(),"val_accuracy"])
    return model

# Custom callback to allow for evaluation of the training set every epoch, without dropout and batch norm (very slow)
'''class EvaluateTrainAndVal(keras.callbacks.Callback):
    
    def __init__(self, train_data):
        super().__init__()
        self.train_data = train_data
        self.train_acc = []

    def on_epoch_end(self, epoch, logs=None):
        # Evaluation mode: dropout OFF, batch norm in inference mode
        _, train_accuracy = self.model.evaluate(
            self.train_data[0], self.train_data[1], verbose=0
        )

        self.train_acc.append(train_accuracy)

        print(
            f"\nEpoch {epoch+1}: "
            f"train_eval_acc={train_accuracy:.4f}, "
        )'''

# Custom tuner to optimize hyperparameters for energy efficiency (FLOPs/MACs)
# using random search for now to develop
class efficiency(keras_tuner.randomSearch):
    def __init__(self, **kwargs):
        # Specify the name of the metric as "custom_metric".
        super().__init__(name="efficiency", **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return
    
    def result(self):
        return
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

model = build_CNN(keras_tuner.HyperParameters()) # set hyperparameters to be optimised

if debug_mode:
    model.summary()

tuner = keras_tuner.RandomSearch(
    build_CNN,
    objective="val_accuracy",
    max_trials=5, # number of hyperparameter combinations to try
    executions_per_trial=5, # number of times to train each model
    directory="tuner_results",
    project_name="mnist_cnn_tuning",
)

#%% Train model

#evaluation = EvaluateTrainAndVal((x_train, y_train))

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=debug_mode)


#%% Evaluate model

score = model.evaluate(x_test, y_test, verbose=debug_mode) # output: [loss, accuracy]

#%% Visualisation

