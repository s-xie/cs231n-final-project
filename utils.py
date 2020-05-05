import pickle
import numpy as np
import tensorflow as tf
from google.cloud import storage

# Constants
IMG_SIZE = 224

# Image Normalization, Resizing, and Augmentation
# modified from https://www.tensorflow.org/tutorials/images/data_augmentation
# Normalize and resize images
def convert(image, label):
	image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])
	image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
	return image, label

# Normalize & resizes images, add random flip, crops, and brightness adjustment
def augment(image,label):
	image,label = convert(image, label)
	image = tf.image.random_flip_left_right(image)
	image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 16, IMG_SIZE + 16) # Add 16 pixels of padding
	image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3]) # Random crop back to 224x224
	image = tf.image.random_brightness(image, max_delta=0.25) # Random brightness
	return image,label

# One hot encode a list of labels y where the labels range from (0, num_classes - 1) inclusive
def one_hot_encode(y, num_classes):
	encoded = np.zeros((len(y), num_classes))
	for idx, val in enumerate(y):
		encoded[idx][val] = 1
	return encoded

# Return GCP Storage bucket for the project
def get_bucket():
	client = storage.Client()
	bucket = client.get_bucket('cs231n-sp2020')
	return bucket

# Download file specified by source_blob_name (stored in GCP bucket bucket) to destination_file_name path on VM
def download_blob(bucket, source_blob_name, destination_file_name):
	blob = bucket.blob(source_blob_name)
	blob.download_to_filename(destination_file_name)


# -------------------------------------- CIFAR 10: will be removed soon ---------------------------------------
# Load training batch file for CIFAR 10 from local machine
def load_cfar10_batch(batch_id):
	with open('../cifar-10-batches-py' + '/data_batch_' + str(batch_id), mode='rb') as file:
		batch = pickle.load(file, encoding='latin1')			
	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']
	return features, labels

# Load training batch file for CIFAR 10 from GCP bucket
def load_cfar10_batch_gcp(batch_id, bucket):
	source_blob_name = 'cifar-10-batches-py/data_batch_' + str(batch_id)
	destination_file_name = '/tmp/batch_' + str(batch_id)
	download_blob(bucket, source_blob_name, destination_file_name)
	with open(destination_file_name, mode = 'rb') as file:
		batch = pickle.load(file, encoding = 'latin1')
	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']
	return features, labels

# Load training and validation datasets for CIFAR 10 from GCP bucket
def get_cfar10_gcp(num_classes):
	train_features = None
	train_labels = None
	bucket = get_bucket()
	for batch_i in range(1, 6):
		features, labels = load_cfar10_batch_gcp(batch_i, bucket)
		encoded_labels = one_hot_encode(labels, num_classes)
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
		test_labels = one_hot_encode(test_labels, num_classes)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
	return train_dataset, test_dataset, num_train_examples

# Load training and validation datasets for CIFAR 10 from local machine
def get_cfar10_local(num_classes):
	train_features = None
	train_labels = None
	for batch_i in range(1, 6):
		features, labels = load_cfar10_batch(batch_i)
		encoded_labels = one_hot_encode(labels, num_classes)
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
		test_labels = one_hot_encode(test_labels, num_classes)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
	return train_dataset, test_dataset, num_train_examples


