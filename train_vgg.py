# Imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
BATCH_SIZE = 64
N_CLASSES = 101
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-d', required = True, choices = ['local', 'gcp-small', 'gcp-large', 'aws-small', 'aws-large'], 
		help = 'local for local data storage, gcp for cloud data storage')
	args = parser.parse_args()
	return args

# Load Training & Validation Data
args = clip()
data_flag = args.d
if data_flag == 'local':
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_local(N_CLASSES)
elif 'gcp' in data_flag:
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_gcp(N_CLASSES, data_flag)
else:
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_aws(N_CLASSES, data_flag)

# Generate batches at tf.data.Dataset objects, applying augmentation to training set
# normalization/resizing applied to both training and validation sets
augmented_train_batches = (
	train_dataset
        .take(-1)
	.cache()
	.shuffle(num_train_examples//4)
	.map(augment, num_parallel_calls=AUTOTUNE)
	.batch(BATCH_SIZE)
	.prefetch(AUTOTUNE)
)

validation_batches = (
	test_dataset
        .take(-1)
        .cache()
	.map(convert, num_parallel_calls=AUTOTUNE)
	.batch(BATCH_SIZE)
)

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
	x = Dense(512, activation = 'relu')(x)
	# can add dropout here later
	out = Dense(N_CLASSES, activation = 'softmax')(x)
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
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 10)

# Compile & train model
model = get_model()
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
model.compile(loss = 'categorical_crossentropy',
							optimizer = adam,
							metrics = [TopKCategoricalAccuracy(k=1, name = 'accuracy'), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])
print('Model compiled')

model_history = model.fit(augmented_train_batches, epochs = 20, validation_data = validation_batches, callbacks = [checkpoint, early_stop])

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
