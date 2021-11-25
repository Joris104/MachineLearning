import numpy as np

def remove_face(pose_sequence):
    sliced_pose_sequence = []
    for pose in pose_sequence:
        pose_part_1 = pose[0:23]
        pose_part_2 = pose[83:]
        sliced_pose = np.concatenate((pose_part_1, pose_part_2))
        sliced_pose_sequence.append(sliced_pose)
    return np.array(sliced_pose_sequence)