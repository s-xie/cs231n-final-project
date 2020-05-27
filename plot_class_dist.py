import numpy as np 
import matplotlib.pyplot as plt 
from utils import *

N_CLASSES = 101

train_dataset, test_dataset, num_train_examples, num_test_examples = get_ucf101_local(N_CLASSES)

class_counts = np.zeros(N_CLASSES)

for element in train_dataset:
	x, y_dist = element
	y = np.argmax(y_dist)
	class_counts[y] += 1

plt.bar(x = range(N_CLASSES), height = class_counts)
plt.xlabel('Class Label Index')
plt.ylabel('Number of Training Videos')
plt.title('Training Set Class Distribution')
plt.savefig('class_dist.png', dpi = 250)