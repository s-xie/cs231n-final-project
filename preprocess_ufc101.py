import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import argparse

def clip():
    parser = argparse.ArgumentParser(description = 'Specify Preprocessing details')
    parser.add_argument('-f', required = True, type = int, help = 'number of frames to sample from each video')
    args = parser.parse_args()
    return args

def get_class_label(f_name):
    l_bound = f_name.index('_')
    r_bound = f_name.index('_', l_bound+1)
    return f_name[l_bound+1:r_bound]

#get all the classes (given as strings) and assign them each an index and create a class_dict
#that maps class string to class index
RAW_DATA_DIR = "ucf101-raw/"
OUTPUT_DATA_DIR = "data/"
TRAIN_SPLIT = "ucfTrainTestlist/trainlist01.txt"
DEV_SPLIT = "ucfTrainTestlist/testlist01.txt"

args = clip()
classes = set()
class_dict = {}

for f_name in glob.glob(RAW_DATA_DIR+"*"):
    class_label = get_class_label(f_name)
    classes.add(class_label)

classes = sorted(list(classes))
for i, c in enumerate(classes):
    class_dict[c] = i  
print(class_dict)

train_files = set()
dev_files = set()

with open(TRAIN_SPLIT, 'r') as f:
    for l in f.readlines():
        p = l.strip().split()[0]
        p = p[p.index('/')+1:]
        train_files.add(p)

with open(DEV_SPLIT, 'r') as f:
    for l in f.readlines():
        p = l.strip().split()[0]
        p = p[p.index('/')+1:]
        dev_files.add(p)

print(dev_files)
count = 0
train_file_num = 1
dev_file_num = 1
X_train, y_train, X_dev, y_dev = [], [], [], []
for f_name in tqdm(glob.glob(RAW_DATA_DIR+"*")):
    count += 1
    vid_name = f_name[f_name.index('/')+1:]
    vidcap = cv2.VideoCapture(f_name)
    num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = num_frames//(args.f - 1)
    count = 0
    success, image = vidcap.read()
    while success:
        if count % sample_rate == 0:
            if vid_name in train_files:
                X_train.append(np.array(image))
                y_train.append(class_dict[get_class_label(f_name)])
            elif vid_name in dev_files:
                X_dev.append(np.array(image))
                y_dev.append(class_dict[get_class_label(f_name)])
            if len(X_train) == 25000:
                np.save(OUTPUT_DATA_DIR + "X_train_" + str(train_file_num), X_train)
                np.save(OUTPUT_DATA_DIR + "y_train_" + str(train_file_num), y_train)
                print('Saved training batch', train_file_num)
                train_file_num += 1
                del X_train[:]
                del y_train[:]
                X_train, y_train = [], []
            if len(X_dev) == 25000:
                np.save(OUTPUT_DATA_DIR + "X_dev_" + str(dev_file_num), X_dev)
                np.save(OUTPUT_DATA_DIR + "y_dev_" + str(dev_file_num), y_dev)
                print('Saved dev batch', dev_file_num)
                dev_file_num += 1
                del X_dev[:]
                del y_dev[:]
                X_dev, y_dev = [], []
        success, image = vidcap.read()
        count += 1
np.save(OUTPUT_DATA_DIR + "X_train_" + str(train_file_num), X_train)
np.save(OUTPUT_DATA_DIR + "y_train_" + str(train_file_num), y_train)
np.save(OUTPUT_DATA_DIR + "X_dev_" + str(dev_file_num), X_dev)
np.save(OUTPUT_DATA_DIR + "y_dev_" + str(dev_file_num), y_dev)


