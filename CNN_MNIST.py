# Benchmarking CNNs on the MNIST dataset for accuracy and energy (FLOPs, MACs)


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
from torch.profiler import profile, ProfilerActivity, record_function # measures actual flops and macs during runtime, depends on batching, optimisers, etc. so more application specific

# CNN
import keras
from keras import layers

# Visualisation

import matplotlib.pyplot as plt

#%% Script options

# Save directory for output plots
SAVE_DIR = './plots/'

# enable to add verbose output and reproducible seeds
debug_mode = True
seed = 1

# CNN parameters
num_layers = 1
num_filters = 2 # number of kernels per unit, this goes for all conv layers
kernel_size = (3, 3) # use an odd number kernel size for your own sake
pool_size = (2, 2) # there will always be one pooling layer per conv layer, set to None to remove pooling layers
activation = "relu"

# Training parameters
batch_size = 128
epochs = 30
optimizers = 
optimizer = "adam"
loss_function = "categorical_crossentropy"

# Data evaluation parameters
evalute_training_set = False # Evaluates using the full training set every epoch without dropout and batch norm giving accurate training accuracy but very slow
#%% Extra function definitions

def build_CNN(input_shape, num_classes, num_layers, num_filters, kernel_size, pool_size = None, activation = "relu"):
    model_params = [keras.Input(shape=input_shape)]

    for _ in range(num_layers):
        model_params.append(layers.Conv2D(num_filters, kernel_size=kernel_size, activation=activation))
        if pool_size is not None:
            model_params.append(layers.MaxPooling2D(pool_size=pool_size))
    model_params.append(layers.Flatten())
    model_params.append(layers.Dropout(0.5))
    model_params.append(layers.Dense(num_classes, activation="softmax"))
    model = keras.Sequential(model_params)
    return model

class EvaluateTrainAndVal(keras.callbacks.Callback):
    
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
        )


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

# split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=debug_mode)

#%% Initialise network
# Adapted from Keras tutorial  https://keras.io/examples/vision/mnist_convnet/

model = build_CNN(input_shape, num_classes, num_layers, num_filters, kernel_size, pool_size, activation)

if debug_mode:
    model.summary()

#%% FLOPs and MACs estimation

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size,) + input_shape,
    output_as_string=True,
)

#%% Compile and train model
'''
evaluation = EvaluateTrainAndVal((x_train, y_train))

model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=debug_mode, callbacks=[evaluation])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=debug_mode) # no callbacks test
'''

#%% test compare gradient descent and Adam

#evaluation = EvaluateTrainAndVal((x_train, y_train))

adam = keras.optimizers.Adam()

simple_gd = keras.optimizers.SGD(momentum=0,netserov=False) #disables stochastic and momentum

optimizers = [adam, simple_gd]

for optimizer in optimizers:
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), verbose=debug_mode)

score = model.evaluate(x_test, y_test, verbose=debug_mode)

#%% Visualisation

plt.figure()
if evalute_training_set:
    plt.plot(evaluation.train_acc,label='Training')
    plt.plot(model.history.history['val_accuracy'],label='Validation')
else:
    plt.plot(model.history.history['val_accuracy'],label = f'filt={num_filters}, dep={num_layers}, kern={kernel_size[0]}, pool={pool_size[0]}, opt={optimizer}')
plt.title('MNIST classification performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
#plt.show()

plt.savefig(SAVE_DIR+'training.svg')