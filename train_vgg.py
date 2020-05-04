import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import pickle
import numpy as np
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE

# will remove once we have actual data
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()

# # Construct a tf.data.Dataset
# dataset, info = tfds.load(name="cifar10", as_supervised = True, with_info = True)
# train_dataset, test_dataset = dataset['train'], dataset['test']
# num_train_examples = info.splits['train'].num_examples

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 25
NUM_EXAMPLES = 100 # Remove for actual training
N_CLASSES = 10 # 101 for actual training
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
	with open('../cifar-10-batches-py' + '/data_batch_' + str(batch_id), mode='rb') as file:
		# note the encoding type is 'latin1'
		batch = pickle.load(file, encoding='latin1')
			
	features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
	labels = batch['labels']
			
	return features, labels

def one_hot_encode(x):
	encoded = np.zeros((len(x), N_CLASSES))
	for idx, val in enumerate(x):
		encoded[idx][val] = 1
	return encoded

train_features = None
train_labels = None
for batch_i in range(1, 6):
	features, labels = load_cfar10_batch('../cifar-10-batches-py', batch_i)
	encoded_labels = one_hot_encode(labels)
	if batch_i == 1:
		train_features = features 
		train_labels = encoded_labels
	else:
		train_features = np.concatenate((train_features, features), axis = 0)
		#print(train_labels.shape)
		#print(labels.shape)
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

# modified from https://www.tensorflow.org/tutorials/images/data_augmentation
def convert(image, label):
	image = tf.image.resize(image, size = [IMG_SIZE, IMG_SIZE])
	image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
	return image, label

def augment(image,label):
	image,label = convert(image, label)
	image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
	image = tf.image.random_flip_left_right(image)
	image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 16, IMG_SIZE + 16) # Add 16 pixels of padding
	image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3]) # Random crop back to 224x224
	image = tf.image.random_brightness(image, max_delta=0.25) # Random brightness
	return image,label

augmented_train_batches = (
		train_dataset
		.take(NUM_EXAMPLES) # change to -1 to get full dataset for actual training
		.cache()
		.shuffle(num_train_examples//4)
		.map(augment, num_parallel_calls=AUTOTUNE)
		.batch(BATCH_SIZE)
		.prefetch(AUTOTUNE)
)

# for element in augmented_train_batches.as_numpy_iterator():
# 	print(element) 
# 	assert False

validation_batches = (
		test_dataset
		.take(NUM_EXAMPLES) # change to -1 to get full dataset for actual training
		.map(convert, num_parallel_calls=AUTOTUNE)
		.batch(BATCH_SIZE)
)

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

model = get_model()
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
model.compile(loss = 'categorical_crossentropy',
							optimizer = adam,
							metrics = [Accuracy(), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])
print('Model compiled')

model_history = model.fit(augmented_train_batches, epochs = 5, validation_data = validation_batches)

plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Model':model_history}, metric = "accuracy")
plt.title("Accuracy")
plt.show()

plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Model':model_history}, metric = "top3_accuracy")
plt.title("Top 3 Accuracy")
plt.show()

plotter = tfdocs.plots.HistoryPlotter()
plotter.plot({'Model':model_history}, metric = "loss")
plt.title("Loss")
plt.show()
