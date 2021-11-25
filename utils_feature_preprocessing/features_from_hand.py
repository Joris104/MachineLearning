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
            hand_base = pose_sequence[FRAME_INDEX][HAND_STARTPOINT]
            for finger_name, indices in FINGER_INDICES.items():
                finger_top = pose_sequence[FRAME_INDEX][HAND_STARTPOINT + indices[-1]]
                dist = np.linalg.norm(hand_base - finger_top)
                extra_features.append(dist)
    # if any(extra_features):
    #     extra_features /= max(extra_features) 
    return extra_features