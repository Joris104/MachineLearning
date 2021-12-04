import numpy as np
from utils_feature_preprocessing.transform_frames import transform_to_k_frames, frames_to_features
from utils_feature_preprocessing.straighten_pose import rotate_seq
from utils_general.utils_for_students import HAND_CONNECTIONS

def pose_hand(pose_sequence):
    left_hands = pose_sequence[:,83:104]
    left_hand_means = np.sum(left_hands, axis=1)/left_hands.shape[1]
    for i in range(0, left_hands.shape[0]):
        left_hands[i] -= left_hands[i, 0]
    
    right_hands = pose_sequence[:,104:]
    right_hand_means = np.sum(right_hands, axis=1)/right_hands.shape[1]
    for i in range(0, right_hands.shape[0]):
        right_hands[i] -= right_hands[i, 0]
    
    left_hands = rotate_seq(left_hands, 0, 1)
    right_hands =  rotate_seq(right_hands, 0, 1)    
    #take the directions


    #rotate
    hand_extended_pose_sequence = []
    
#     hand_extended_pose_sequence = transform_to_k_frames(np.array(hand_extended_pose_sequence), k=1)
    left = transform_to_k_frames(left_hands, k=1)
    right = transform_to_k_frames(right_hands, k=1)
    connections = []
    for connection in HAND_CONNECTIONS:
        connections.append(left[0, connection[0]] - left[0,connection[1]])
        connections.append(right[0, connection[0]] - right[0,connection[1]])

    return frames_to_features(connections)