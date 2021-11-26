import numpy as np

#def translate_body(pose_sequence):
#    #translates pose sequence so that middle between shoulders starts at the same position in the first frame
#    center_between_shoulders = np.sum((pose_sequence[:, 11] + pose_sequence[:, 12]))/(pose_sequence.shape[0]*2)
#    return pose_sequence - center_between_shoulders

def translate_shoulder_to_y(pose):
    #translates the poses center between the two shoulders to be on the y-axis
    center_between_shoulders = (pose[11] + pose[12])/2
    return pose - center_between_shoulders

def translate_seq_shoulder_to_y(pose_sequence):
    translated_sequence = []
    for pose in pose_sequence:
        translated_sequence.append(translate_shoulder_to_y(pose))
    return np.array(translated_sequence)

def rotate_to_parallel_xy(pose):
    #Rotate the pose based on the leftshoulder positions (landmark 11, leftshoulder)
    #So that the shoulders are parallel with the xy plane
    #IMPORTANT: the function assumes that the center between landmark 11 and 12 is on the y axis
    #First calculate necessary corner theta for rotation 
    left_shoulder = pose[11] / np.linalg.norm(pose[11])
    theta = np.arccos(np.clip(np.dot((left_shoulder[0],left_shoulder[1],0), left_shoulder), -1.0, 1.0))
    
    #Then create the rotation matrix, take transpose for easier calculation in next step
    rotation_matrix_transpose = np.transpose(np.array([[np.cos(theta), 0, np.sin(theta)],
                                                       [0, 1, 0],
                                                       [-np.sin(theta), 0, np.cos(theta)]]))
    
    #finally rotat all keypoints in the pose
    pose = pose @ rotation_matrix_transpose
    return pose

def rotate_seq_to_parallel_xy(pose_sequence):
    #rotates all poses in a sequence
    rotated_pose_sequence = []
    for pose in pose_sequence:
        rotated_pose_sequence.append(rotate_to_parallel_xy(pose))
    return np.array(rotated_pose_sequence)

def straighten_pose_seq(pose_sequence):
    pose_sequence = translate_seq_shoulder_to_y(pose_sequence)
    pose_sequence = rotate_seq_to_parallel_xy(pose_sequence)
    return pose_sequence