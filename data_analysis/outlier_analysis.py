import matplotlib.pyplot as plt
import numpy as np
import os

import utils_for_students

from vectors import to_vector


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
