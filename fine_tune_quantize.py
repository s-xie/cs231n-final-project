# NOTE: REQUIRES TENSORFLOW-NIGHTLY!
# Imports
import argparse 
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from utils import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
N_CLASSES = 101
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4
EPOCHS = 5

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-d', required = True, choices = [ 'aws-small', 'aws-large'], 
		help = 'local for local data storage, gcp or aws for cloud data storage')
	parser.add_argument('-m', required = True, type = str, help = 'path to model file')
	args = parser.parse_args()
	return args

def freeze_layers(model):
	print(len(model.layers))
	for i, layer in enumerate(model.layers):
		if i < N_LAYERS_TO_FREEZE:
			layer.trainable = False 
		else:
			layer.trainable = True

# Load Training & Validation Data
args = clip()
train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_aws_generator(N_CLASSES, args.d)
X_train, y_train = train_dataset
X_dev, y_dev = test_dataset

train_gen = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.2, height_shift_range = 0.1, rescale = 1./255, 
								horizontal_flip = True, fill_mode = 'nearest')
dev_gen = ImageDataGenerator(rescale = 1./255)

# Load model
model_path = args.m
model = load_model(filepath = model_path, compile = True)
print('Model Loaded!')

def apply_quantization_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

q_aware_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_dense,
)
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
q_aware_model.compile(optimizer = adam, loss = 'categorical_crossentropy', 
						metrics = [TopKCategoricalAccuracy(k=1, name = 'accuracy'), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])

# Define callbacks
model_folder = os.path.dirname(model_path)
model_filename = os.path.basename(model_path)
output_filename = model_filename[:model_filename.index('.hdf5')] + '_quant.hdf5'
checkpoint = ModelCheckpoint(filepath = os.path.join(model_folder, output_filename), save_best_only = True,
							monitor = 'val_accuracy', save_weights_only = False, verbose = 0)
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 20)

# Fine-tune model
#model_history = q_aware_model.fit(augmented_train_batches, epochs = 5, validation_data = validation_batches, callbacks = [checkpoint, early_stop])
model_history = q_aware_model.fit(train_gen.flow(X_train, y_train, shuffle = True, batch_size = BATCH_SIZE),
									steps_per_epoch = num_train_examples // BATCH_SIZE, epochs = EPOCHS,
									validation_data = dev_gen.flow(X_dev, y_dev, shuffle = True, batch_size = BATCH_SIZE),
									validation_steps = num_test_examples // BATCH_SIZE, callbacks = [checkpoint, early_stop])

# Save output
plot_folder = os.path.join(model_folder, 'plots/')
np.save(os.path.join(model_folder, 'history_quant.npy'), model_history.history)
metrics = ['loss', 'accuracy', 'top3_accuracy', 'top5_accuracy']
titles = ['Loss', 'Accuracy', 'Top 3 Accuracy', 'Top 5 Accuracy']
for i, metric in enumerate(metrics):
	train_metric = model_history.history[metric]
	val_metric = model_history.history['val_' + metric]
	plt.figure()
	epochs = range(len(train_metric))
	plt.plot(epochs, train_metric, label = 'Train')
	plt.plot(epochs, val_metric, label = 'Validation')
	plt.title(titles[i] + ' Over Time - Quantize-Aware Fine-Tuning')
	plt.xlabel('Fine-Tuning Epochs')
	plt.ylabel(titles[i])
	plt.legend()
	plt.savefig(os.path.join(plot_folder, metric + '_quant_fine_tune.png'), dpi = 200)
