# Imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
from google.cloud import storage
from utils import *
import argparse
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Constants & Parameters
OUTPUT_DIR = "models/"
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
N_CLASSES = 101
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-d', required = True, choices = ['aws-small', 'aws-large'], 
		help = 'local for local data storage, gcp or aws for cloud data storage')
	args = parser.parse_args()
	return args

# Load Training & Validation Data
args = clip()
train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_aws_generator(N_CLASSES, args.d)
X_train, y_train = train_dataset
X_dev, y_dev = test_dataset

train_gen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.1, rescale = 1./255, 
								horizontal_flip = True, fill_mode = 'nearest')
dev_gen = ImageDataGenerator(rescale = 1./255)

# Define and load model
# borrowed from https://github.com/nnbenavides/Fine-Grained-Vehicle-Classification/blob/master/train_vgg_model.py
def freeze_layers(model):
	print(len(model.layers))
	for i, layer in enumerate(model.layers):
		if i < N_LAYERS_TO_FREEZE:
			layer.trainable = False 
		else:
			layer.trainable = True

def add_layers(model):
	x = model.output
	x = Flatten()(x)
	x = Dropout(0.8)(x)
	x = Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(1e-4))(x)
	x = Dropout(0.8)(x)
	out = Dense(N_CLASSES, activation = 'softmax', kernel_regularizer = regularizers.l2(1e-4))(x)
	model = Model(inputs = model.input, outputs = out)
	print('VGG16 model loaded and modified!')
	return model

def get_model():
	base_model = tf.keras.applications.vgg16.VGG16(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
	#print(base_model.summary()) # 5 conv blocks, 14.7M total params, 14.7M trainable params
	freeze_layers(base_model)
	#print(base_model.summary()) # 5 conv blocks, 14.7M total params, 2.4M trainable params
	model = add_layers(base_model)
	print(model.summary()) # 5 conv blocks + 2 FC layers, 27.6M total params, 15.2M trainable params
	return model

# Make sure we have an output folder for model files, history, and plots
if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)
model_folder = os.path.join(OUTPUT_DIR, 'model_' + str(strftime("%Y-%m-%d_%H-%M-%S", gmtime())))
if not os.path.exists(model_folder):
	os.mkdir(model_folder)
plot_folder = os.path.join(model_folder, 'plots/')
if not os.path.exists(plot_folder):
	os.mkdir(plot_folder)

# Define callbacks
checkpoint = ModelCheckpoint(filepath = os.path.join(model_folder, 'model.hdf5'), save_best_only = True,
							monitor = 'val_accuracy', save_weights_only = False, verbose = 0)
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 5)

# Compile & train model
model = get_model()
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
model.compile(loss = 'categorical_crossentropy',
							optimizer = adam,
							metrics = [TopKCategoricalAccuracy(k=1, name = 'accuracy'), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])
print('Model compiled')

model_history = model.fit(train_gen.flow(X_train, y_train, shuffle = True, batch_size = BATCH_SIZE),
									steps_per_epoch = num_train_examples // BATCH_SIZE, epochs = 18,
									validation_data = dev_gen.flow(X_dev, y_dev, shuffle = True, batch_size = BATCH_SIZE),
									validation_steps = num_test_examples // BATCH_SIZE, callbacks = [checkpoint, early_stop])

# Save output
np.save(os.path.join(model_folder, 'history.npy'), model_history.history)
metrics = ['loss', 'accuracy', 'top3_accuracy', 'top5_accuracy']
titles = ['Loss', 'Accuracy', 'Top 3 Accuracy', 'Top 5 Accuracy']
for i, metric in enumerate(metrics):
	train_metric = model_history.history[metric]
	val_metric = model_history.history['val_' + metric]
	plt.figure()
	epochs = range(len(train_metric))
	plt.plot(epochs, train_metric, label = 'Train')
	plt.plot(epochs, val_metric, label = 'Validation')
	plt.title(titles[i] + ' Over Time')
	plt.xlabel('Epochs')
	plt.ylabel(titles[i])
	plt.legend()
	plt.savefig(os.path.join(plot_folder, metric + '.png'), dpi = 200)
