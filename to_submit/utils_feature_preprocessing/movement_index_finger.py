import numpy as np
from utils_feature_preprocessing.transform_frames import transform_to_k_frames, frames_to_features, select_frames
from utils_feature_preprocessing.pose_hands import drop_unknown_hands

def movement_index_finger(pose_sequence):
	begin_pose = select_frames(pose_sequence, 0, 2)
	end_pose = select_frames(pose_sequence, -1, -2)

	position_face = begin_pose[0, 0]
	movement_index_left = end_pose[0, 90]-begin_pose[0, 90]
	movement_index_right = end_pose[0, 111]-begin_pose[0, 111]

	return np.array([movement_index_left, movement_index_right]).flatten()