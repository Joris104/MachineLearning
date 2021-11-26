import numpy as np
from utils_feature_preprocessing.transform_frames import transform_to_k_frames, frames_to_features

def pose_hand(pose_sequence):
    hand_extended_pose_sequence = []
    for pose in pose_sequence:
        left_hand = pose[83:104]
        left_hand_mean = np.sum(left_hand, axis=0)/left_hand.shape[0]
        left_hand -= left_hand_mean
        right_hand = pose[104:]
        right_hand_mean = np.sum(right_hand, axis=0)/right_hand.shape[0]
        right_hand -= right_hand_mean
        #We take the sum because some movements can be done with the left or right hand. This way we 
        #assure that the features are consistent within one class
        hand_extended_pose = np.sum([left_hand, right_hand], axis=0)/2
        hand_extended_pose_sequence.append(hand_extended_pose)
    hand_extended_pose_sequence = transform_to_k_frames(np.array(hand_extended_pose_sequence), k=1)
    hand_features = frames_to_features(hand_extended_pose_sequence)

    return hand_features