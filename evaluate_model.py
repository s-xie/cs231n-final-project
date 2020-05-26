import argparse
import tensorflow as tf 
from tensorflow.keras.models import load_model
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import os
import matplotlib.pyplot as plt
from utils import *
AUTOTUNE = tf.data.experimental.AUTOTUNE

N_CLASSES = 101
ERROR_ANALYSIS_EXAMPLES = 200

def clip():
	parser = argparse.ArgumentParser(description = 'Specify evaluation details')
	parser.add_argument('-d', required = True, choices = ['local', 'gcp-small', 'gcp-large', 'aws-small', 'aws-large'], 
		help = 'local for local data storage, gcp or aws for cloud data storage')
	parser.add_argument('-b', required = True, type = int, help = 'batch size (number of frames per video)')
	parser.add_argument('-train', action = 'store_true', help = 'flag to specify if you want to evaluate training set or not')
	parser.add_argument('-test', action = 'store_true', help = 'flag to specify if you want to evaluate test set or not')
	parser.add_argument('-m', required = True, type = str, help = 'path to model file')
	parser.add_argument('-err', action = 'store_true', help = 'flag to specify if you want to store some misclassifications for error analysis')
	args = parser.parse_args()
	return args

# Load Training & Validation Data
args = clip()
data_flag = args.d
error_flag = args.err
if data_flag == 'local':
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_local(N_CLASSES)
elif 'gcp' in data_flag:
	train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_gcp(N_CLASSES, data_flag)
else:
	if args.test and args.train:
		train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_aws(N_CLASSES, data_flag)
	elif args.test:
		_ , test_dataset, _ , num_test_examples = get_ucf101_aws(N_CLASSES, data_flag, 'test')
	elif args.train:
		train_dataset, _, num_train_examples, num_test_examples = get_ucf101_aws(N_CLASSES, data_flag, 'train')

model = load_model(filepath = args.m, compile = False)
print('Model Loaded!')

batch_size = args.b

def evaluate_model(model, dataset, num_examples, batch_size, mode, model_path, error_flag):
	if mode == 'train':
		print('Evaluating Training Set...')
	elif mode == 'test':
		print('Evaluating Test Set...')

	num_batches = num_examples // batch_size
	preds_mode = np.zeros(num_batches)
	preds_avg = np.zeros(num_batches)
	actual_classes = np.empty(num_batches)
	preds_avg_top5 = np.zeros((num_batches, 5))
	
	if error_flag:
		error_images = []
		error_pred_classes = []
		error_actual_classes = []
		error_pred_classes_video = []
		error_counter = 0
		class_dict = np.load('class_dict.npy', allow_pickle = True).item()
		reversed_dict = {v: k for k, v in class_dict.items()}

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

		# Error analysis w/ reservoir sampling
		if error_flag and (preds_avg[counter] != actual_classes[counter]):
			if len(error_images) < ERROR_ANALYSIS_EXAMPLES:
				error_images.append(batch_x)
				error_actual_classes.append(actual_classes[counter])
				error_pred_classes.append(pred_classes)
				error_pred_classes_video.append(preds_avg[counter])
				error_counter += 1
			else:
				replace = np.random.rand() < ((ERROR_ANALYSIS_EXAMPLES - 1)/error_counter)
				if replace:
					rand_idx = np.random.randint(0, ERROR_ANALYSIS_EXAMPLES)
					error_images[rand_idx] = batch_x 
					error_actual_classes[rand_idx] = actual_classes[counter]
					error_pred_classes[rand_idx] = pred_classes
					error_pred_classes_video[rand_idx] = preds_avg[counter]
				error_counter += 1

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
	model_filename = os.path.basename(model_path)
	eval_filename = model_filename[:model_filename.index('.hdf5')] + '_eval.txt'
	eval_file = os.path.join(model_folder, eval_filename)
	with open(eval_file, mode = 'a+') as out_file:
		if mode == 'train':
			out_file.write('Training Set Results:\n')
		elif mode == 'test':
			out_file.write('Test Set Results:\n')
		out_file.write('Accuracy (Mode Voting Scheme): ' + str(mode_acc) + '\n')
		out_file.write('Accuracy (Average Voting Scheme): ' + str(avg_acc) + '\n')
		out_file.write('Top 5 Accuracy (Average Voting Scheme): ' + str(avg_acc_top5) + '\n')
		out_file.write('\n')

	if error_flag:
		error_folder = os.path.join(model_folder, 'errors_' + model_filename + '/')
		if not os.path.exists(error_folder):
			os.mkdir(error_folder)
		for i, images in enumerate(error_images):
			example_folder = str(i) + ' - pred = ' + reversed_dict[int(error_pred_classes_video[i])] + ' - actual = ' + reversed_dict[int(error_actual_classes[i])] + '/'
			example_folder = os.path.join(error_folder, example_folder)
			os.mkdir(example_folder)
			for j in range(batch_size):
				image = images[j,:,:,:]
				img_filename = str(j) + ' - pred = ' + reversed_dict[int(error_pred_classes[i][j])] + '.jpg'
				img_filepath = os.path.join(example_folder, img_filename)
				plt.imsave(img_filepath, image)

if args.test:
	validation_batches = (
		test_dataset
		.take(50)
		.cache()
		.map(convert, num_parallel_calls=AUTOTUNE)
		.batch(batch_size)
	)
	evaluate_model(model, validation_batches, num_test_examples, batch_size, 'test', args.m, error_flag)
	
if args.train:
	train_batches = (
		train_dataset
		.take(-1)
		.cache()
		.map(convert, num_parallel_calls=AUTOTUNE)
		.batch(batch_size)
	)
	evaluate_model(model, train_batches, num_train_examples, batch_size, 'train', args.m, False)