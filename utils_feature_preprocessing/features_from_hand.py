import numpy as np

def extract_finger_distances(pose_sequence):
    FINGER_INDICES = {
        "THUMB":list(range(1,5)), 
        "INDEX":list(range(5,9)),
        "MIDDLE":list(range(9,13)),
        "RING":list(range(13,17)),
        "PINKY":list(range(17,21)),
    }
    HAND_STARTPOINTS = [83,104]
    extra_features = []
    for HAND_STARTPOINT in HAND_STARTPOINTS:
        for FRAME_INDEX in range(pose_sequence.shape[0]):
            features_per_hand = []
            hand_wrist = pose_sequence[FRAME_INDEX][HAND_STARTPOINT]
            hand_knuckles = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + FINGER_INDICES["MIDDLE"][0]]
            dist_wrist_knuckles = np.linalg.norm(hand_wrist - hand_knuckles)
            for finger_name, indices in FINGER_INDICES.items():
                finger_top = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + indices[-1]]
                dist_wrist_top = np.linalg.norm(hand_wrist - finger_top)
                features_per_hand.append(dist_wrist_top)
            if any(features_per_hand):
                features_per_hand /= dist_wrist_knuckles
            extra_features.extend(features_per_hand)
    extra_features = np.floor(extra_features)
    return extra_features

def extract_average_hands(pose_sequence):
    HAND_STARTPOINTS = [83,104]
    HAND_COUNTPOINTS = 21
    extra_features = []
    for HAND_STARTPOINT in HAND_STARTPOINTS:
        for FRAME_INDEX in range(pose_sequence.shape[0]):
            mean_position = np.mean(pose_sequence[FRAME_INDEX][HAND_STARTPOINT:HAND_STARTPOINT+HAND_COUNTPOINTS], axis=0)
            extra_features.append(mean_position)
    extra_features = np.stack(extra_features).reshape(-1)
    return extra_features
