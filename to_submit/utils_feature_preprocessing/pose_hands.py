import numpy as np
from utils_feature_preprocessing.transform_frames import transform_to_k_frames, frames_to_features
from utils_feature_preprocessing.straighten_pose_old import rotate_seq
from utils_general.utils_for_students import HAND_CONNECTIONS
from utils_feature_preprocessing.transform_frames import select_frames

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
    left1 = transform_to_k_frames(left_hands, k=1)
    right1 = transform_to_k_frames(right_hands, k=1)
    left = transform_to_k_frames(left_hands, k=2)
    right = transform_to_k_frames(right_hands, k=2)


    connections = []
    for connection in HAND_CONNECTIONS:
        connections.append(left1[0, connection[0]] - left1[0,connection[1]])
        connections.append(right1[0, connection[0]] - right1[0,connection[1]])
        connections.append(left[0, connection[0]] - left[0,connection[1]])
        connections.append(right[0, connection[0]] - right[0,connection[1]])
        connections.append(left[1, connection[0]] - left[1,connection[1]])
        connections.append(right[1, connection[0]] - right[1,connection[1]])

    #we now look at how the hands have moved throughout the video:
   # mov_left = left_end-left_begin
   # mov_right = right_end-right_begin

    connections = frames_to_features(connections)

    return connections#np.concatenate((connections, mov_left, mov_right))