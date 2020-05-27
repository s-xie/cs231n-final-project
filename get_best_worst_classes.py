import numpy as np
import argparse

def clip():
	parser = argparse.ArgumentParser(description = 'Specify error analysis file details')
	parser.add_argument('-p', required = True, type = str, help = 'path to class accuracies file')
	args = parser.parse_args()
	return args

args = clip()

# Read in class dict
class_dict = np.load('class_dict.npy', allow_pickle = True).item()
reversed_dict = {v: k for k, v in class_dict.items()}

# read in lines from class accuracies file
with open(args.p, mode = 'r') as input_file:
	lines = input_file.readlines()

class_accs = [float(line.split()[-1]) for line in lines]
ordered_indices = list(np.argsort(class_accs))

worst_ten_indices = ordered_indices[:10]
best_ten_indices = ordered_indices[-10:]
best_ten_indices.reverse()

print('Worst 10 Class Accuracies')
for i, idx in enumerate(worst_ten_indices):
	print(str(i+1) + '. ' + reversed_dict[idx] + ': ' + str(round(class_accs[idx], 2)) + '%')

print('\n')

print('Best 10 Class Accuracies')
for i, idx in enumerate(best_ten_indices):
	print(str(i+1) + '. ' + reversed_dict[idx] + ': ' + str(round(class_accs[idx], 2)) + '%')
