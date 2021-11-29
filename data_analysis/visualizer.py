from utils_general.utils_for_students import label_encoder
from utils_general import utils_for_students
import os
import matplotlib.pyplot as plt
import numpy as np
def visualizer(l1, l2, train_samples, y_train, y_pred):
	K = 4
	CLASSES = label_encoder().inverse_transform([i for i in range(len(label_encoder().classes_))])
	LABEL = l1
	LABEL2 = l2
	# Extract for specific label
	indices, = np.where((y_train == LABEL) & (y_pred==LABEL2))
	print("Showing", len(indices) ,"labels of class",CLASSES[LABEL],"predicted as",CLASSES[LABEL2])

	for idx in indices:
		sample = train_samples[idx]
		seq = utils_for_students.load_sample_stage2(os.path.join('data/stage2/train/', sample['path']))
		fig, axs = plt.subplots(1, len(seq),figsize=(30,5))
		for i in range(len(seq)):
			axis = axs[i]
			utils_for_students.visualize_pose(seq[i], axis)

	#for idx in indices_true[6:7]:
	#    sample = train_samples[idx]
	#    seq = utils_for_students.load_sample_stage2(os.path.join('data/stage2/train/', sample['path']))
	#    fig, axs = plt.subplots(1, len(seq),figsize=(30,5))
	#    for i in range(len(seq)):
	#        axis = axs[i]
	#        utils_for_students.visualize_pose(seq[i], axis)