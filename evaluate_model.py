import argparse
import tensorflow as tf 
from tensorflow.keras.models import load_model
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import os
from utils import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

N_CLASSES = 10 # 101 for UCF

def clip():
	parser = argparse.ArgumentParser(description = 'Specify evaluation details')
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
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_cfar10_local(N_CLASSES)
else:
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_cfar10_gcp(N_CLASSES)

model = load_model(filepath = args.m, compile = False)
print('Model Loaded!')

batch_size = args.b

def evaluate_model(model, dataset, num_examples, batch_size, mode, model_path):
	if mode == 'train':
		print('Evaluating Training Set...')
	elif mode == 'test':
		print('Evaluating Test Set...')

	num_batches = num_examples // batch_size
	preds_mode = np.zeros(num_batches)
	preds_avg = np.zeros(num_batches)
	actual_classes = np.empty(num_batches)
	preds_avg_top5 = np.zeros((num_batches, 5))

	counter = 0
	for element in tqdm(dataset.as_numpy_iterator()):
		batch_x = element[0]
		batch_y = element[1]
		actual_classes[counter] = np.argmax(batch_y[0]) # all elements are the same since each frame in a video has same label
		preds = np.asarray(model.predict_on_batch(batch_x))

		# Average voting scheme
		avg_pred = preds.mean(axis = 0)
		preds_avg[counter] = avg_pred.argmax()
		top5_preds = top_n_indices(avg_pred, 5)
		preds_avg_top5[counter,:] = top5_preds

		# Mode voting scheme
		pred_classes = preds.argmax(axis = 1)
		preds_mode[counter] = stats.mode(pred_classes)[0][0]
		counter += 1

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
	eval_file = os.path.join(model_folder, 'eval.txt')
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
		.cache()
		.map(convert, num_parallel_calls=AUTOTUNE)
		.batch(batch_size)
	)
	evaluate_model(model, validation_batches, num_test_examples, batch_size, 'test', args.m)
	
if args.train:
	train_batches = (
		train_dataset
		.cache()
		.map(convert, num_parallel_calls=AUTOTUNE)
		.batch(batch_size)
	)
	evaluate_model(model, train_batches, num_train_examples, batch_size, 'train', args.m)