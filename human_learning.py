
import csv
import pickle
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from utils_general import utils_for_students
train_samples = utils_for_students.load_dataset_stage2('data/stage2_labels_train.csv', 'train')

i = []
for idx, sample in enumerate(train_samples):
    if sample['label'] == 'c.OOK' or sample['label'] == 'ZELFDE-A':
        i.append(idx)

np.random.shuffle(i)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

corr = 0
for j in range(10):
    print(f"Sequence {j+1} of 10")
    sample = train_samples[i[j]]
    seq = utils_for_students.load_sample_stage2(os.path.join('data/stage2/train/', sample['path']))
    
    for s in seq:
        ax.set_xlim(0.3,0.7)
        ax.set_ylim(-0.5,0.9)
        utils_for_students.visualize_pose(s, ax)
        plt.draw()
        plt.pause(0.2)
        plt.cla()
    ans = input("OOK or ZELFDE ?\n").strip()
    if ans in sample['label']:
        print("Correct")
        corr += 1
    else :
        print(f"Fout - het was {sample['label']}")
    time.sleep(2)

print(f"Accuracy : {corr*10}%")
