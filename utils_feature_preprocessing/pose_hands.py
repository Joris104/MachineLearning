import numpy as np
from utils_feature_preprocessing.transform_frames import transform_to_k_frames, frames_to_features, select_frames
from utils_feature_preprocessing.straighten_pose import rotate_seq
from utils_general.utils_for_students import HAND_CONNECTIONS

def drop_unknown_hands(hands):
    filtered = []
    for hand in hands:
        if np.all(hand != 0):
            filtered.append(hand)
    if (len(filtered) != 0):
        return np.array(filtered)
    else:
        return hands

def pose_hand(pose_sequence):

    left_hands = pose_sequence[:,83:104]
    #If left hand is still not present in a frame we ignore it (we take an average thus 0 can influence the results)
    left_hands = drop_unknown_hands(left_hands)
    left_hands = rotate_seq(left_hands, 0, 1)

    right_hands = pose_sequence[:,104:]
    #If right hand is still not present in a frame we ignore it (we take an average thus 0 can influence the results)
    right_hands = drop_unknown_hands(right_hands)
    right_hands =  rotate_seq(right_hands, 0, 1)    
    
    #take the directions
    left = transform_to_k_frames(left_hands, k=1)
    right = transform_to_k_frames(right_hands, k=1)
    connections = []
    for connection in HAND_CONNECTIONS:
        connections.append(left[0, connection[0]] - left[0,connection[1]])
        connections.append(right[0, connection[0]] - right[0,connection[1]])

    #also take first couple of frames to average and last couple of frames
    left_begin = select_frames(left_hands, 0, 3) # 0 1 2
    left_end = select_frames(left_hands, -1, -3) # -3 -2 -1
    right_begin = select_frames(right_hands, 0, 3)
    right_end = select_frames(right_hands, -1, -3)

    #we now look at how the hands have moved throughout the video:
    mov_left = left_end-left_begin
    mov_right = right_end-right_begin

    connections = frames_to_features(connections)
    mov_left = frames_to_features(mov_left)
    mov_right = frames_to_features(mov_right)

    return connections#np.concatenate((connections, mov_left, mov_right))