import numpy as np

#def translate_body(pose_sequence):
#    #translates pose sequence so that middle between shoulders starts at the same position in the first frame
#    center_between_shoulders = np.sum((pose_sequence[:, 11] + pose_sequence[:, 12]))/(pose_sequence.shape[0]*2)
#    return pose_sequence - center_between_shoulders

def translate_shoulders(pose):
    #translates the poses center between the two shoulders to be on the y-axis
    #center_between_shoulders = (pose[11] + pose[12])/2
    return pose - pose[12]#center_between_shoulders

def translate_seq_shoulders(pose_sequence):
    translated_sequence = []
    for pose in pose_sequence:
        translated_sequence.append(translate_shoulders(pose))
    return np.array(translated_sequence)

def rotate_to_parallel_xy(pose, vec):
    #Rotate the pose based on the leftshoulder positions (landmark 11, leftshoulder)
    if np.all(vec != 0):
        point = vec / np.linalg.norm(vec)
    else:
        point = vec
    theta = np.arccos(np.clip(point[0], -1.0, 1.0))
    #Then create the rotation matrix, take transpose for easier calculation in next step
    rotation_matrix_transpose = np.transpose(np.array([[np.cos(theta), 0, np.sin(theta)],
                                                       [0, 1, 0],
                                                       [-np.sin(theta), 0, np.cos(theta)]]))

    #finally rotate all keypoints in the pose
    pose = pose @ rotation_matrix_transpose
    return pose

def rotate_to_parallel_xz(pose, vec):
    if np.all(vec != 0):
        point = vec / np.linalg.norm(vec)
    else:
        point = vec
    theta = np.arccos(np.clip(np.dot((0,1,0), (point[0],point[1], 0)), -1.0, 1.0))
    #Then create the rotation matrix, take transpose for easier calculation in next step
    rotation_matrix_transpose = np.transpose(np.array([[np.cos(theta), -np.sin(theta), 0],
                                                       [np.sin(theta), np.cos(theta), 0],
                                                       [0, 0, 1]]))
    
    #finally rotate all keypoints in the pose
    pose = pose @ rotation_matrix_transpose
    return pose

def rotate_seq(pose_sequence, alginment_point_1, alignment_point_2):
    #rotates all poses in a sequence
    rotated_pose_sequence = []
    for pose in pose_sequence:
        vec = pose[alginment_point_1]-pose[alignment_point_2]
        new_pose = rotate_to_parallel_xy(pose, vec)
        new_pose = rotate_to_parallel_xz(new_pose, vec)
        rotated_pose_sequence.append(new_pose)
    return np.array(rotated_pose_sequence)

def straighten_pose_seq(pose_sequence):
    pose_sequence = translate_seq_shoulders(pose_sequence)
    pose_sequence = rotate_seq(pose_sequence, 11, 12)
    return pose_sequence