# Imports
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from time import gmtime, strftime
from google.cloud import storage
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Constants & Parameters
DATA_DIR = ""
OUTPUT_DIR = "models/"
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 64
NUM_EXAMPLES = 1280 # Remove for actual training
N_CLASSES = 10 # 101 for actual training
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4

# Load Training & Validation Data
def load_cfar10_batch(batch_id):
	with open('../cifar-10-batches-py' + '/data_batch_' + str(batch_id), mode='rb') as file:
		# note the encoding type is 'latin1'
		batch = pickle.load(file, encoding='latin1')
			
	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']
			
	return features, labels

def download_blob(bucket, source_blob_name, destination_file_name):
	blob = bucket.blob(source_blob_name)
	blob.download_to_filename(destination_file_name)

def load_cfar10_batch_gcp(batch_id, bucket):
	source_blob_name = 'cifar-10-batches-py/data_batch_' + str(batch_id)
	destination_file_name = '/tmp/batch_' + str(batch_id)
	download_blob(bucket, source_blob_name, destination_file_name)
	with open(destination_file_name, mode = 'rb') as file:
		batch = pickle.load(file, encoding = 'latin1')
	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']
	return features, labels

def one_hot_encode(x):
	encoded = np.zeros((len(x), N_CLASSES))
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	return encoded

def get_cfar10_gcp():
	train_features = None
	train_labels = None
	client = storage.Client()
	bucket = client.get_bucket('cs231n-sp2020')
	for batch_i in range(1, 6):
		features, labels = load_cfar10_batch_gcp(batch_i, bucket)
		encoded_labels = one_hot_encode(labels)
		if batch_i == 1:
			train_features = features 
			train_labels = encoded_labels
		else:
			train_features = np.concatenate((train_features, features), axis = 0)
			train_labels = np.concatenate((train_labels, encoded_labels), axis = 0)
	print(train_features.shape)
	print(train_labels.shape)
	num_train_examples = train_features.shape[0]

	test_features, test_labels = None, None
	source_blob_name = 'cifar-10-batches-py/test_batch'
	destination_file_name = '/tmp/test_batch'
	download_blob(bucket, source_blob_name, destination_file_name)
	with open(destination_file_name, mode='rb') as file:
		batch = pickle.load(file, encoding='latin1')
		# preprocess the testing data
		test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
		test_labels = batch['labels']
		test_labels = one_hot_encode(test_labels)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
	return train_dataset, test_dataset, num_train_examples

def get_cfar10_local():
	train_features = None
	train_labels = None
	for batch_i in range(1, 6):
		features, labels = load_cfar10_batch(batch_i)
		encoded_labels = one_hot_encode(labels)
		if batch_i == 1:
			train_features = features 
			train_labels = encoded_labels
		else:
			train_features = np.concatenate((train_features, features), axis = 0)
			train_labels = np.concatenate((train_labels, encoded_labels), axis = 0)
	print(train_features.shape)
	print(train_labels.shape)
	num_train_examples = train_features.shape[0]

	test_features, test_labels = None, None
	with open('../cifar-10-batches-py' + '/test_batch', mode='rb') as file:
		batch = pickle.load(file, encoding='latin1')

		# preprocess the testing data
		test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
		test_labels = batch['labels']
		test_labels = one_hot_encode(test_labels)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
	return train_dataset, test_dataset, num_train_examples

train_dataset, test_dataset, num_train_examples = get_cfar10_gcp()

# Image Normalization, Resizing, and Augmentation
# modified from https://www.tensorflow.org/tutorials/images/data_augmentation
def convert(image, label):
	image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])
	image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
	return image, label

def augment(image,label):
	image,label = convert(image, label)
	image = tf.image.random_flip_left_right(image)
	image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 16, IMG_SIZE + 16) # Add 16 pixels of padding
	image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3]) # Random crop back to 224x224
	image = tf.image.random_brightness(image, max_delta=0.25) # Random brightness
	return image,label

# Generate batches at tf.data.Dataset objects
augmented_train_batches = (
		train_dataset
		.take(NUM_EXAMPLES) # change to -1 to get full dataset for actual training
		.cache()
		.shuffle(num_train_examples//4)
		.map(augment, num_parallel_calls=AUTOTUNE)
		.batch(BATCH_SIZE)
		.prefetch(AUTOTUNE)
)

validation_batches = (
		test_dataset
		.take(NUM_EXAMPLES) # change to -1 to get full dataset for actual training
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
model_folder = os.path.join(OUTPUT_DIR, 'model_' + str(strftime("%Y-%m-%d_%H:%M:%S", gmtime())))
if not os.path.exists(model_folder):
	os.mkdir(model_folder)
plot_folder = os.path.join(model_folder, 'plots/')
if not os.path.exists(plot_folder):
	os.mkdir(plot_folder)

# Define callbacks
checkpoint = ModelCheckpoint(filepath = os.path.join(model_folder, 'model.hdf5'), save_best_only = True,
							monitor = 'val_accuracy', save_weights_only = False, verbose = 0)
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 10)
#tb = TensorBoard(log_dir = LOG_DIR, batch_size = BATCH_SIZE)

# Compile & train model
model = get_model()
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
model.compile(loss = 'categorical_crossentropy',
							optimizer = adam,
							metrics = [Accuracy(), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])
print('Model compiled')

model_history = model.fit(augmented_train_batches, epochs = 5, validation_data = validation_batches, callbacks = [checkpoint, early_stop])

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
