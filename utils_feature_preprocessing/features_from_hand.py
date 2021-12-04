import numpy as np

FINGER_INDICES = {
    "THUMB":list(range(1,5)), 
    "INDEX":list(range(5,9)),
    "MIDDLE":list(range(9,13)),
    "RING":list(range(13,17)),
    "PINKY":list(range(17,21)),
}
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


def extract_finger_curviness(pose_sequence):
    extra_features = []
    # For each frame
    for FRAME_INDEX in range(pose_sequence.shape[0]):
        # For both hands
        features_per_frame = []
        for HAND_STARTPOINT in HAND_STARTPOINTS:
            features_per_hand = []
            wrist_point = pose_sequence[FRAME_INDEX][HAND_STARTPOINT]
            # For every finger
            for finger_name, indices in FINGER_INDICES.items():
                # Calculate length of finger (from wrist) by looping over points
                start_point = wrist_point
                finger_dist = 0
                for joint_index in indices:
                    joint_point = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + joint_index]
                    finger_dist += np.linalg.norm(joint_point - start_point)
                    start_point = joint_point
                # Calculate distance from wrist to top of finger
                wrist_to_top_dist = np.linalg.norm(wrist_point - start_point) # start_point is top now
                if finger_dist < 0.0001:
                    features_per_hand.append(0) 
                else:
                    features_per_hand.append(wrist_to_top_dist/finger_dist)
            features_per_frame.extend(features_per_hand)
        extra_features.append(features_per_frame)
    return np.array(extra_features)

def extract_average_hands(pose_sequence):
    HAND_STARTPOINTS = [83,104]
    HAND_COUNTPOINTS = 21
    extra_features = []
    for FRAME_INDEX in range(pose_sequence.shape[0]):
        features_per_frame = []
        for HAND_STARTPOINT in HAND_STARTPOINTS:
            mean_position = np.mean(pose_sequence[FRAME_INDEX][HAND_STARTPOINT:HAND_STARTPOINT+HAND_COUNTPOINTS], axis=0)
            features_per_frame.extend(mean_position)
        extra_features.append(features_per_frame)
    return np.array(extra_features)

def extract_average_arms(pose_sequence):
    extra_features = []
    for FRAME_INDEX in range(pose_sequence.shape[0]):
        features_per_frame = []
        for indices in [[13,15],[14,16]]:
            elbow = np.array(pose_sequence[FRAME_INDEX][indices[0]])
            wrist = np.array(pose_sequence[FRAME_INDEX][indices[1]])
            features_per_frame.extend((wrist + elbow)/2)
        extra_features.append(features_per_frame)
    return extra_features


def extract_arm_orientations(pose_sequence):
    extra_features = []
    for FRAME_INDEX in range(pose_sequence.shape[0]):
        features_per_frame = []
        for indices in [[13,15],[14,16]]:
            elbow = pose_sequence[FRAME_INDEX][indices[0]]
            wrist = pose_sequence[FRAME_INDEX][indices[1]]
            features_per_frame.extend(_get_orientation(elbow, wrist))
        extra_features.append(features_per_frame)
    return extra_features

def extract_wrist_orientations(pose_sequence):
    extra_features = []
    for FRAME_INDEX in range(pose_sequence.shape[0]):
        features_per_frame = []
        for indices in [[15,17,19],[16,18,20]]:
            wrist = pose_sequence[FRAME_INDEX][indices[0]]
            knuckles = np.mean([pose_sequence[FRAME_INDEX][indices[1]],pose_sequence[FRAME_INDEX][indices[2]]], axis=0)
            features_per_frame.extend(_get_orientation(wrist, knuckles))
        extra_features.append(features_per_frame)
    return extra_features


def _get_orientation(start_point, end_point):
    diff_vector = np.array(end_point) - np.array(start_point)
    norm = np.linalg.norm(diff_vector)
    if norm == 0:
        return diff_vector
    return diff_vector / norm


# def extract_finger_distances(pose_sequence):
#     extra_features = []
#     for HAND_STARTPOINT in HAND_STARTPOINTS:
#         for FRAME_INDEX in range(pose_sequence.shape[0]):
#             features_per_hand = []
#             hand_wrist = pose_sequence[FRAME_INDEX][HAND_STARTPOINT]
#             hand_knuckles = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + FINGER_INDICES["MIDDLE"][0]]
#             dist_wrist_knuckles = np.linalg.norm(hand_wrist - hand_knuckles)
#             for finger_name, indices in FINGER_INDICES.items():
#                 finger_top = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + indices[-1]]
#                 dist_wrist_top = np.linalg.norm(hand_wrist - finger_top)
#                 features_per_hand.append(dist_wrist_top)
#             if any(features_per_hand):
#                 features_per_hand /= dist_wrist_knuckles
#             extra_features.extend(features_per_hand)
#     return extra_features