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
import os
import matplotlib.pyplot as plt
from utils import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 25
NUM_EXAMPLES = 100 # Remove for actual training
N_CLASSES = 101
N_LAYERS_TO_FREEZE = 17 # freeze everything before the last conv layer
lr = 1e-4

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-d', required = True, choices = ['local', 'gcp'], 
		help = 'local for local data storage, gcp for cloud data storage')
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
model_history = model.fit(augmented_train_batches, epochs = 5, validation_data = validation_batches, callbacks = [checkpoint, early_stop])

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