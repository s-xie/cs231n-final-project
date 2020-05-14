import pickle
import numpy as np
import tensorflow as tf
from google.cloud import storage
from smart_open import smart_open

# Constants
IMG_SIZE = 224

# Image Normalization, Resizing, and Augmentation
# modified from https://www.tensorflow.org/tutorials/images/data_augmentation
# Normalize and resize images
def convert(image, label):
	#image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])
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
		encoded[idx][int(val)] = 1
	return encoded

# Returns the indices corresponding to the n largest values of arr
def top_n_indices(arr, n):
	idx = (-arr).argsort()[:n]
	return idx

# Returns the top n accuracy of a set of predictions
# y_true - [num_examples, ] numpy array of true classes
#y_pred - [num_examples, n] numpy array of top n predicted classes
def top_n_accuracy(y_true, y_pred):
	num_correct = 0
	num_examples = y_true.shape[0]
	for i in range(y_true.shape[0]):
		if y_true[i] in list(y_pred[i,:]):
			num_correct += 1
	return num_correct/num_examples

# Return GCP Storage bucket for the project
def get_bucket():
	client = storage.Client()
	bucket = client.get_bucket('cs231n-sp2020')
	return bucket

# Download file specified by source_blob_name (stored in GCP bucket bucket) to destination_file_name path on VM
def download_blob(bucket, source_blob_name, destination_file_name):
	blob = bucket.blob(source_blob_name)
	blob.download_to_filename(destination_file_name)

def get_ucf101_local(num_classes):
	data_dir = 'data/'
	num_train_files = 2
	num_dev_files = 1

	X_train, y_train = None, None
	print('Loading training files (' + str(num_train_files) + ' batch(es)) ...')
	for i in range(1, num_train_files + 1):
		with open(data_dir + 'X_train_' + str(i) + '.npy', mode = 'rb') as file:
			batch = np.load(file, allow_pickle = True)
		with open(data_dir + 'y_train_' + str(i) + '.npy', mode = 'rb') as file:
			y = np.load(file, allow_pickle = True)
			y = one_hot_encode(y, num_classes)
		if i == 1:
			X_train = batch
			y_train = y
		else:
			X_train = np.concatenate((X_train, batch), axis = 0)
			y_train = np.concatenate((y_train, y), axis = 0)
		print('Loaded training batch ' + str(i))
	print('Shape of X_train:', X_train.shape)
	print('Shape of y_train:', y_train.shape)

	X_dev, y_dev = None, None
	print('Loading dev files (' + str(num_dev_files) + ' batch(es)) ...')
	for i in range(1, num_dev_files + 1):
			with open(data_dir + 'X_dev_' + str(i) + '.npy', mode = 'rb') as file:
				batch = np.load(file, allow_pickle = True)
			with open(data_dir + 'y_dev_' + str(i) + '.npy', mode = 'rb') as file:
				y = np.load(file, allow_pickle = True)
				y = one_hot_encode(y, num_classes)
			if i == 1:
				X_dev = batch
				y_dev = y
			else:
				X_dev = np.concatenate((X_dev, batch), axis = 0)
				y_dev = np.concatenate((y_dev, y), axis = 0)
			print('Loaded dev batch ' + str(i))
	print('Shape of X_dev:', X_dev.shape)
	print('Shape of y_dev:', y_dev.shape)

	num_train_examples = X_train.shape[0]
	num_dev_examples = X_dev.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	print('Generated training tf.data.dataset')
	dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev))
	print('Generated dev tf.data.dataset')
	return train_dataset, dev_dataset, num_train_examples, num_dev_examples

def get_ucf101_aws(num_classes, mode):
	if mode == 'aws-small':
		num_train_files = 2
		num_dev_files = 1
		data_dir = 'ucf101-small/'
		s3_prefix = 's3://cs231n-bucket/ucf101-small/'
	else:
		num_train_files = 8
		num_dev_files = 4
		data_dir = 'ucf101-large/'
		s3_prefix = 's3://cs231n-bucket/ucf101-small/'

	X_train, y_train = None, None
	print('Loading training files (' + str(num_train_files) + ' batch(es)) ...') 
	for i in range(1, num_train_files + 1):
		# Get X_train_i
		#source_blob_name = data_dir + 'X_train_' + str(i) + '.npy'
		#destination_file_name = '/tmp/X_train_' + str(i) + '.npy'
		#download_blob(bucket, source_blob_name, destination_file_name)
		with smart_open(s3_prefix + 'X_train_' + str(i) + '.npy', mode = 'rb') as file:
			batch = np.load(file, allow_pickle = True)
		# Get y_train_i
		#source_blob_name = data_dir + 'y_train_' + str(i) + '.npy'
		#destination_file_name = '/tmp/y_train_' + str(i) + '.npy'
		#download_blob(bucket, source_blob_name, destination_file_name)
		with open(s3_prefix + 'y_train_' + str(i) + '.npy', mode = 'rb') as file:
			y = np.load(file, allow_pickle = True)
			y = one_hot_encode(y, num_classes)
		if i == 1:
			X_train = batch
			y_train = y
		else:
			X_train = np.concatenate((X_train, batch), axis = 0)
			y_train = np.concatenate((y_train, y), axis = 0)
		print('Loaded training batch ' + str(i))
	print('Shape of X_train:', X_train.shape)
	print('Shape of y_train:', y_train.shape)

	X_dev, y_dev = None, None
	print('Loading dev files (' + str(num_dev_files) + ' batch(es)) ...')
	for i in range(1, num_dev_files + 1):
		# Get X_dev_i
		#source_blob_name = data_dir + 'X_dev_' + str(i) + '.npy'
		#destination_file_name = '/tmp/X_dev_' + str(i) + '.npy'
		#download_blob(bucket, source_blob_name, destination_file_name)
		with open(s3_prefix + 'X_dev_' + str(i) + '.npy', mode = 'rb') as file:
			batch = np.load(file, allow_pickle = True)
		# Get y_dev_i
		#source_blob_name = data_dir + 'y_dev_' + str(i) + '.npy'
		#destination_file_name = '/tmp/y_dev_' + str(i) + '.npy'
		#download_blob(bucket, source_blob_name, destination_file_name)
		with open(s3_prefix + 'y_dev_' + str(i) + '.npy', mode = 'rb') as file:
			y = np.load(file, allow_pickle = True)
			y = one_hot_encode(y, num_classes)
		if i == 1:
			X_dev = batch
			y_dev = y
		else:
			X_dev = np.concatenate((X_dev, batch), axis = 0)
			y_dev = np.concatenate((y_dev, y), axis = 0)
		print('Loaded dev batch ' + str(i))
	print('Shape of X_dev:', X_dev.shape)
	print('Shape of y_dev:', y_dev.shape)

	num_train_examples = X_train.shape[0]
	num_dev_examples = X_dev.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	print('Generated training tf.data.dataset')
	dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev))
	print('Generated dev tf.data.dataset')
	return train_dataset, dev_dataset, num_train_examples, num_dev_examples

def get_ucf101_gcp(num_classes, mode):
	if mode == 'gcp-small':
		num_train_files = 2
		num_dev_files = 1
		data_dir = 'ucf-small/'
	else:
		num_train_files = 8
		num_dev_files = 4
		data_dir = 'ucf-large/'
	bucket = get_bucket()

	X_train, y_train = None, None
	print('Loading training files (' + str(num_train_files) + ' batch(es)) ...') 
	for i in range(1, num_train_files + 1):
		# Get X_train_i
		source_blob_name = data_dir + 'X_train_' + str(i) + '.npy'
		destination_file_name = '/tmp/X_train_' + str(i) + '.npy'
		download_blob(bucket, source_blob_name, destination_file_name)
		with open(destination_file_name, mode = 'rb') as file:
			batch = np.load(file, allow_pickle = True)
		# Get y_train_i
		source_blob_name = data_dir + 'y_train_' + str(i) + '.npy'
		destination_file_name = '/tmp/y_train_' + str(i) + '.npy'
		download_blob(bucket, source_blob_name, destination_file_name)
		with open(destination_file_name, mode = 'rb') as file:
			y = np.load(file, allow_pickle = True)
			y = one_hot_encode(y, num_classes)
		if i == 1:
			X_train = batch
			y_train = y
		else:
			X_train = np.concatenate((X_train, batch), axis = 0)
			y_train = np.concatenate((y_train, y), axis = 0)
		print('Loaded training batch ' + str(i))
	print('Shape of X_train:', X_train.shape)
	print('Shape of y_train:', y_train.shape)

	X_dev, y_dev = None, None
	print('Loading dev files (' + str(num_dev_files) + ' batch(es)) ...')
	for i in range(1, num_dev_files + 1):
		# Get X_dev_i
		source_blob_name = data_dir + 'X_dev_' + str(i) + '.npy'
		destination_file_name = '/tmp/X_dev_' + str(i) + '.npy'
		download_blob(bucket, source_blob_name, destination_file_name)
		with open(destination_file_name, mode = 'rb') as file:
			batch = np.load(file, allow_pickle = True)
		# Get y_dev_i
		source_blob_name = data_dir + 'y_dev_' + str(i) + '.npy'
		destination_file_name = '/tmp/y_dev_' + str(i) + '.npy'
		download_blob(bucket, source_blob_name, destination_file_name)
		with open(destination_file_name, mode = 'rb') as file:
			y = np.load(file, allow_pickle = True)
			y = one_hot_encode(y, num_classes)
		if i == 1:
			X_dev = batch
			y_dev = y
		else:
			X_dev = np.concatenate((X_dev, batch), axis = 0)
			y_dev = np.concatenate((y_dev, y), axis = 0)
		print('Loaded dev batch ' + str(i))
	print('Shape of X_dev:', X_dev.shape)
	print('Shape of y_dev:', y_dev.shape)

	num_train_examples = X_train.shape[0]
	num_dev_examples = X_dev.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
	print('Generated training tf.data.dataset')
	dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, y_dev))
	print('Generated dev tf.data.dataset')
	return train_dataset, dev_dataset, num_train_examples, num_dev_examples

# -------------------------------------- CIFAR 10: will be removed soon ---------------------------------------
# Load training batch file for CIFAR 10 from local machine
# def load_cfar10_batch(batch_id):
# 	with open('../cifar-10-batches-py' + '/data_batch_' + str(batch_id), mode='rb') as file:
# 		batch = pickle.load(file, encoding='latin1')			
# 	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
# 	labels = batch['labels']
# 	return features, labels

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

	num_test_examples = test_features.shape[0]
	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
	return train_dataset, test_dataset, num_train_examples, num_test_examples

# Load training and validation datasets for CIFAR 10 from local machine
# def get_cfar10_local(num_classes):
# 	train_features = None
# 	train_labels = None
# 	for batch_i in range(1, 6):
# 		features, labels = load_cfar10_batch(batch_i)
# 		encoded_labels = one_hot_encode(labels, num_classes)
# 		if batch_i == 1:
# 			train_features = features 
# 			train_labels = encoded_labels
# 		else:
# 			train_features = np.concatenate((train_features, features), axis = 0)
# 			train_labels = np.concatenate((train_labels, encoded_labels), axis = 0)
# 	print(train_features.shape)
# 	print(train_labels.shape)
# 	num_train_examples = train_features.shape[0]

# 	test_features, test_labels = None, None
# 	with open('../cifar-10-batches-py' + '/test_batch', mode='rb') as file:
# 		batch = pickle.load(file, encoding='latin1')

# 		# preprocess the testing data
# 		test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
# 		test_labels = batch['labels']
# 		test_labels = one_hot_encode(test_labels, num_classes)

# 	num_test_examples = test_features.shape[0]
# 	train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
# 	test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
# 	return train_dataset, test_dataset, num_train_examples, num_test_examples
