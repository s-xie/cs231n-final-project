import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# import matplotlib.pyplot as plt


def get_class_label(f_name):
    l_bound = f_name.index('_')
    r_bound = f_name.index('_', l_bound+1)
    return f_name[l_bound+1:r_bound]

#get all the classes (given as strings) and assign them each an index and create a class_dict
#that maps class string to class index
RAW_DATA_DIR = "rawData/"
DATA_DIR = "data/"
classes = set()
class_dict = {}

for f_name in glob.glob(RAW_DATA_DIR+"*"):
    class_label = get_class_label(f_name)
    classes.add(class_label)

classes = sorted(list(classes))
for i, c in enumerate(classes):
    class_dict[c] = i  
print(class_dict)

count = 0
X, y = [], []
for f_name in tqdm(glob.glob(RAW_DATA_DIR+"*")):
    count += 1
    if count % 100 == 0:
        time.sleep(1)
    vidcap = cv2.VideoCapture(f_name)
    success,image = vidcap.read()

    while success:
        X.append(np.array(image))
        y.append(class_dict[get_class_label(f_name)])
        success,image = vidcap.read()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1, stratify = y)
X_dev, X_test, y_dev, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1, stratify = y_val)

np.save(DATA_DIR + "X_train", X_train)
np.save(DATA_DIR + "y_train", y_train)
np.save(DATA_DIR + "X_dev", X_dev)
np.save(DATA_DIR + "y_dev", y_dev)
np.save(DATA_DIR + "X_test", X_test)
np.save(DATA_DIR + "y_test", y_test)


