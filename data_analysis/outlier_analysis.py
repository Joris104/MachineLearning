import matplotlib.pyplot as plt
import numpy as np
import os,sys
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils_general import utils_for_students
from utils_feature_preprocessing.vectors import to_vector


data = {}

train_samples = utils_for_students.load_dataset_stage2('data/stage2_labels_train.csv', 'train')
for sample in train_samples:
    keypoints = utils_for_students.load_sample_stage2(os.path.join('data/stage2/train/', sample['path']))
    vectors = to_vector(keypoints)

    lens = np.sqrt(np.sum(np.square(vectors), axis=2))
    lens.reshape(vectors.shape[0] * vectors.shape[1])

    if not sample['label'] in data:
        data[sample['label']] = np.array([])
    data[sample['label']] = np.append(data[sample['label']], lens)

for label, lens in data.items():
    plt.title(label)
    plt.hist(lens)
    plt.show()
