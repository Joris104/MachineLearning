import numpy as np

HAND_STARTPOINTS = [83,104]
BODY_INDICES = {
    "NOSE": [0],
    "LEFT_BROW": [1,2,3,7],
    "RIGHT_BROW": [4,5,6,8],
    "MOUTH": [9,10],
    "LEFT_SHOULDER_ELBOW_WRIST": [11,13,15],
    "RIGHT_SHOULDER_ELBOW_WRIST": [12,14,16],
    "LEFT_HAND": [17,19,21],
    "RIGHT_HAND": [18,20,22],
}


def is_body_part_missing(frame, include_body = True, include_face = True, include_lhand = True, include_rhand = True):
    if len(frame.shape) > 2:
        raise ValueError("parameter should be frame (not a pose_sequence)")
    body = np.linalg.norm(frame[:23]) == 0
    face = np.linalg.norm(frame[23:83]) == 0
    lhand = np.linalg.norm(frame[83:104]) == 0
    rhand = np.linalg.norm(frame[104:125]) == 0
    return (include_body and body, include_face and face, include_lhand and lhand, include_rhand and rhand)


def get_distance_hands_on_wrists(frame):
    if len(frame.shape) > 2:
        raise ValueError("parameter should be frame (not a pose_sequence)")
    HAND_WRISTS = [83,104]
    BODY_WRISTS = [15,16]
    dist_vect_left = frame[HAND_WRISTS[0]] - frame[BODY_WRISTS[0]]
    dist_vect_right = frame[HAND_WRISTS[1]] - frame[BODY_WRISTS[2]]
    return np.linalg.norm(dist_vect_left), np.linalg.norm(dist_vect_right)


def is_hand_not_on_wrist(frame, threshold = 0.05):
    if len(frame.shape) > 2:
        raise ValueError("parameter should be frame (not a pose_sequence)")
    dist_left, dist_right = get_distance_hands_on_wrists(frame)
    return max(dist_left - threshold, dist_right - threshold) > 0


def is_bad_sample(pose_sequence):
    # A missing body part -> bad sample
    for frame in pose_sequence:
        if any(is_body_part_missing(frame, include_face=False)):
            return True
        if is_hand_not_on_wrist(frame):
            return True
    return False
    