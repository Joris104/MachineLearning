import numpy as np
import csv
import pickle
import time
import os
import matplotlib.pyplot as plt

import utils_for_students
from utils_feature_preprocessing import split_features_into_frames


def to_vector(keypoints):
    #returns the vectors, of length one less than incoming keypoints
    l = len(keypoints)
    vector = np.empty((l-1,125,3))
    for i in range(1,l):
        vector[i-1] = keypoints[i] - keypoints[i-1]
    return vector
    
def extract_features(pose_sequence):
    NUM_SLICES = 2 #TODO : inspect how performance changes with more slices
    size = len(pose_sequence[0])*len(pose_sequence[0][0]) # number of keypoints * number of values per keypoint
    pose_sequence = split_features_into_frames(pose_sequence, NUM_SLICES)
    poses = np.array_split(pose_sequence,NUM_SLICES) #some of these may be empty
    features = poses[0] #save the initial position
    #here you cut to much information, in the middle of the video we are not necessarily on the same position
    for i in range(1, NUM_SLICES):
        vector = poses[i] - poses[i-1]
        features  = np.append(features, vector)
    return features


def plt_log10_hist(hist, labels, title):
  fig, axs = plt.subplots(1, 3, figsize=(13,9),sharey=True)
  XYZ = ["X","Y","Z"]
  for i in range(3):
    for h, l in zip(hist, labels):
        axs[i].hist(np.log10(h[:,i]),[-6,-5,-4,-3,-2,-1,0], density = True, alpha=0.5, label=l)
        axs[i].set_title(XYZ[i])
  fig.suptitle(title)
  axs[0].set_ylabel("Relative frequency")
  plt.legend(loc='upper right')
  plt.show()

def movFromVectors(vectors):
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
    return body_movement, face_movement, lhand_movement, rhand_movement
        
def movFromFeatures(vectors):
    frame = vectors[375:]
    body_movement = frame[:23*3]
    face_movement = frame[23*3:83*3]
    lhand_movement = frame[83*3:104*3]
    rhand_movement   = frame[104*3:]
    
    return body_movement, face_movement, lhand_movement, rhand_movement
if __name__ == "__main__":
  label_analysis = False
  label = "SCHILDPAD-Bhanden" #label to study
  body_hist = []
  face_hist = []
  hand_hist = []
  train_samples = utils_for_students.load_dataset_stage2('data/stage2_labels_train.csv', 'train')
  for sample in train_samples:
    if label_analysis and not sample["label"] == label:
      continue
    keypoints = utils_for_students.load_sample_stage2(os.path.join('data/train/', sample['path']))
    vectors = to_vector(keypoints)
    #print(vectors.shape)
    #BODY from 0 to 23
    #face from 23 to 83
    #left hand from 83 to 104
    #right hand from 104 to 125
    #body_movement, face_movement, lhand_movement, rhand_movement = movFromFeatures(vectors)
    body_movement, face_movement, lhand_movement, rhand_movement = movFromVectors(vectors)
    face_hist.append(face_movement)
    body_hist.append(body_movement)
    hand_hist.append(lhand_movement)
    hand_hist.append(rhand_movement)
  face_hist = np.vstack(np.vstack(face_hist))
  hand_hist = np.vstack(np.vstack(hand_hist))
  body_hist = np.vstack(np.vstack(body_hist))
  
  labels = ["Hand", "Body"]
  plt_log10_hist([hand_hist, body_hist], labels , "Movement per frame (Log Scale)")
  plt_log10_hist([face_hist], ["Face"], "Face movement per frame (Log scale)")
  #plt_log10_hist(body_hist, "Body movement")
