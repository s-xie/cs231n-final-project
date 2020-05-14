# Imports
import tensorflow as tf
import argparse
import numpy as np 
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import accuracy_score
import os
from utils import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

N_CLASSES = 101

def clip():
	parser = argparse.ArgumentParser(description = 'Specify TFLite evaluation details')
	parser.add_argument('-d', required = True, choices = ['local', 'gcp'], 
		help = 'local for local data storage, gcp for cloud data storage')
	parser.add_argument('-b', required = True, type = int, help = 'batch size (number of frames per video)')
	parser.add_argument('-train', action = 'store_true', help = 'flag to specify if you want to evaluate training set or not')
	parser.add_argument('-test', action = 'store_true', help = 'flag to specify if you want to evaluate test set or not')
	parser.add_argument('-m', required = True, type = str, help = 'path to model file')
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

interpreter = tf.lite.Interpreter(model_path = args.m)
interpreter.allocate_tensors()
print('TFLite Model Loaded!')

batch_size = args.b

def evaluate_tflite_model(model, dataset, num_examples, batch_size, mode, model_path):
	if mode == 'train':
		print('Evaluating Training Set...')
	elif mode == 'test':
		print('Evaluating Test Set...')

	num_batches = num_examples // batch_size
	preds_mode = np.zeros(num_batches)
	preds_avg = np.zeros(num_batches)
	actual_classes = np.empty(num_batches)
	preds_avg_top5 = np.zeros((num_batches, 5))

	# set up interpreter
	input_index = interpreter.get_input_details()[0]['index']
	output_index = interpreter.get_output_details()[0]['index']

	batch_num = 0
	examples_in_batch = 0
	for image, label in tqdm(dataset): # need to iterate over each image
		if examples_in_batch == 0: # first image in batch fo frames for 1 video
			actual_classes[batch_num] = np.argmax(label) # all elements are the same since each frame in a video has same label
			preds = np.zeros((batch_size, N_CLASSES))
		
		input_image = np.expand_dims(image, axis = 0).astype(np.float32)

		# run inference on the TFLite model
		interpreter.set_tensor(input_index, input_image)
		interpreter.invoke()
		pred = interpreter.get_tensor(output_index)
		preds[examples_in_batch,:] = pred

		if examples_in_batch == batch_size - 1: # last image in batch
			# Average voting scheme
			avg_pred = preds.mean(axis = 0)
			preds_avg[batch_num] = avg_pred.argmax()
			top5_preds = top_n_indices(avg_pred, 5)
			preds_avg_top5[batch_num,:] = top5_preds

			# Mode voting scheme
			pred_classes = preds.argmax(axis = 1)
			preds_mode[batch_num] = stats.mode(pred_classes)[0][0]
			
			batch_num += 1 # increment to next batch/video
			examples_in_batch = 0 #reset for next batch/video
		else: # not the last image in the batch/video
			examples_in_batch += 1

	mode_acc = accuracy_score(actual_classes, preds_mode)
	avg_acc = accuracy_score(actual_classes, preds_avg)
	avg_acc_top5 = top_n_accuracy(actual_classes, preds_avg_top5)

	if mode == 'train':
		print('Training Set Results:')
	elif mode == 'test':
		print('Test Set Results:')
	print('Accuracy (Mode Voting Scheme):', mode_acc)
	print('Accuracy (Average Voting Scheme):', avg_acc)
	print('Top 5 Accuracy (Average Voting Scheme):', avg_acc_top5)

	model_folder = os.path.dirname(model_path)
	model_filename = os.path.basename(model_path)
	eval_file = os.path.join(model_folder, model_filename[:model_filename.index('.tflite')] + '_eval.txt')
	with open(eval_file, mode = 'a+') as out_file:
		if mode == 'train':
			out_file.write('Training Set Results:\n')
		elif mode == 'test':
			out_file.write('Test Set Results:\n')
		out_file.write('Accuracy (Mode Voting Scheme): ' + str(mode_acc) + '\n')
		out_file.write('Accuracy (Average Voting Scheme): ' + str(avg_acc) + '\n')
		out_file.write('Top 5 Accuracy (Average Voting Scheme): ' + str(avg_acc_top5) + '\n')
		out_file.write('\n')

if args.test:
	validation_batches = (
		test_dataset
		.map(convert, num_parallel_calls=AUTOTUNE)
	)
	evaluate_tflite_model(interpreter, validation_batches, num_test_examples, batch_size, 'test', args.m)
	
if args.train:
	train_batches = (
		train_dataset
		.map(convert, num_parallel_calls=AUTOTUNE)
	)
	evaluate_model(model, train_batches, num_train_examples, batch_size, 'train', args.m)