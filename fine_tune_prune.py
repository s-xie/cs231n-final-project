# NOTE: REQUIRES TENSORFLOW-NIGHTLY!
# ref: https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
# Imports
import argparse 
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.metrics import TopKCategoricalAccuracy, Accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow_model_optimization.sparsity import keras as sparsity
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
EPOCHS = 10
sparsity_target = 0.7 # indicates final sparsity target for pruning; between 0 and 1

def clip():
	parser = argparse.ArgumentParser(description = 'Specify training details')
	parser.add_argument('-d', required = True, choices = ['aws-small', 'aws-large'], 
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

def apply_pruning_to_dense(layer):
  end_step = np.ceil(1.0 * num_train_examples / BATCH_SIZE).astype(np.int32) * EPOCHS
  pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                   final_sparsity=sparsity_target,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
  }
  if isinstance(layer, tf.keras.layers.Dense):
    return sparsity.prune_low_magnitude(layer, **pruning_params) # note: can pass a whole model instead of individual layers to prune_low_magnitude if desired
  return layer

pruned_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_dense,
)
adam = optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2 = 0.99, epsilon = None, decay = 1e-5, amsgrad = False)
pruned_model.compile(optimizer = adam, loss = 'categorical_crossentropy', 
						metrics = [TopKCategoricalAccuracy(k=1, name = 'accuracy'), TopKCategoricalAccuracy(k = 3, name = 'top3_accuracy'), TopKCategoricalAccuracy(k = 5, name = 'top5_accuracy')])

# Define callbacks
model_folder = os.path.dirname(model_path)
model_filename = os.path.basename(model_path)
output_filename = model_filename[:model_filename.index('.hdf5')] + '_prune' + str(sparsity_target) + '.hdf5'
checkpoint = ModelCheckpoint(filepath = os.path.join(model_folder, output_filename), save_best_only = False,
							monitor = 'val_accuracy', save_weights_only = False, verbose = 0)
#early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 20)
pruning_step = sparsity.UpdatePruningStep()
log_folder = os.path.join(model_folder, 'logs/')
if not os.path.exists(log_folder):
	os.mkdir(log_folder)
print('Writing training logs to ' + log_folder)
pruning_summary = sparsity.PruningSummaries(log_dir=log_folder, profile_batch=0)

# Fine-tune model
#model_history = model.fit(augmented_train_batches, epochs = 5, validation_data = validation_batches, callbacks = [checkpoint, early_stop, pruning_step, pruning_summary])
model_history = pruned_model.fit(train_gen.flow(X_train, y_train, shuffle = True, batch_size = BATCH_SIZE),
									steps_per_epoch = num_train_examples // BATCH_SIZE, epochs = EPOCHS,
									validation_data = dev_gen.flow(X_dev, y_dev, shuffle = True, batch_size = BATCH_SIZE),
									validation_steps = num_test_examples // BATCH_SIZE, callbacks = [checkpoint, pruning_step, pruning_summary]) #took out early_stop

pruned_output_filename = output_filename[:-5] + '_comp.hdf5'
export_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
tf.keras.models.save_model(export_model, pruned_output_filename, include_optimizer=False)

# Save output
plot_folder = os.path.join(model_folder, 'plots/')
np.save(os.path.join(model_folder, 'history_prune.npy'), model_history.history)
metrics = ['loss', 'accuracy', 'top3_accuracy', 'top5_accuracy']
titles = ['Loss', 'Accuracy', 'Top 3 Accuracy', 'Top 5 Accuracy']
for i, metric in enumerate(metrics):
	train_metric = model_history.history[metric]
	val_metric = model_history.history['val_' + metric]
	plt.figure()
	epochs = range(len(train_metric))
	plt.plot(epochs, train_metric, label = 'Train')
	plt.plot(epochs, val_metric, label = 'Validation')
	plt.title(titles[i] + ' Over Time - Pruned Fine-Tuning')
	plt.xlabel('Fine-Tuning Epochs')
	plt.ylabel(titles[i])
	plt.legend()
	plt.savefig(os.path.join(plot_folder, metric + '_prune_fine_tune.png'), dpi = 200)
