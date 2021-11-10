import numpy as np
import csv
import pickle
import time
import os
import matplotlib.pyplot as plt

import utils_for_students

def extract_features(pose_sequence):
    NUM_SLICES = 2 #TODO : inspect how performance changes with more slices
    size = len(pose_sequence[0])*len(pose_sequence[0][0]) # number of keypoints * number of values per keypoint
    poses = np.array_split(pose_sequence,NUM_SLICES) #some of these may be empty
    features = pose_sequence[0] #save the initial position
    for p in poses:
        if len(p) == 0:
            vector = np.zeros(size)
        else:
            vector = p[-1] - p[0]
        features  = np.append(features, vector)
    features = features.reshape(-1)
    features[abs(features) > 0.2] = 0.0
    return features

def to_vector(keypoints):
    #returns the vectors, of length one less than incoming keypoints
    l = len(keypoints)
    vector = np.empty((l-1,125,3))
    for i in range(1,l):
        vector[i-1] = keypoints[i] - keypoints[i-1]
    return vector

def plt_log10_hist(hist, title):
  fig, axs = plt.subplots(1, 3, figsize=(13,9),sharey=True)
  XYZ = ["X","Y","Z"]
  for i in range(3):
    axs[i].hist(np.log10(hist[:,i]),[-6,-5,-4,-3,-2,-1,0])
    axs[i].set_title(XYZ[i])
  fig.suptitle(title)
  plt.show()
    
if __name__ == "__main__":
  label = "SCHILDPAD-Bhanden" #label to study
  body_hist = []
  face_hist = []
  lhand_hist = []
  rhand_hist = []
  train_samples = utils_for_students.load_dataset_stage2('data/stage2_labels_train.csv', 'train')
  for sample in train_samples:
    if not sample["label"] == label:
      continue
    keypoints = utils_for_students.load_sample_stage2(os.path.join('data/train/', sample['path']))
    vectors = to_vector(keypoints)
    l = len(vectors)
    body_movement = np.empty((l,23,3))
    face_movement = np.empty((l,60,3))
    lhand_movement = np.empty((l,21,3))
    rhand_movement = np.empty((l,21,3))
    #BODY from 0 to 23
    #face from 23 to 83
    #left hand from 83 to 104
    #right hand from 104 to 125
    for i,frame in zip(range(l),vectors):
        body_movement[i] = frame[:23,:]
        face_movement[i] = frame[23:83,:]
        lhand_movement[i] = frame[83:104,:]
        rhand_movement[i] = frame[104:,:]
    face_hist.append(face_movement)
    body_hist.append(body_movement)
    lhand_hist.append(lhand_movement)
  face_hist = np.vstack(np.vstack(face_hist))
  lhand_hist = np.vstack(np.vstack(lhand_hist))
  body_hist = np.vstack(np.vstack(body_hist))
  
  plt_log10_hist(face_hist, "Face movement")
  plt_log10_hist(lhand_hist, "Hand movement")
  plt_log10_hist(body_hist, "Body movement")
