import numpy as np

def is_body_part_missing(frame, include_body = True, include_face = True, include_lhand = True, include_rhand = True):
    if len(frame.shape) > 2:
        raise ValueError("parameter should be frame (not a pose_sequence)")
    body = np.linalg.norm(frame[:23]) == 0
    face = np.linalg.norm(frame[23:83]) == 0
    lhand = np.linalg.norm(frame[83:104]) == 0
    rhand = np.linalg.norm(frame[104:125]) == 0
    return (include_body and body, include_face and face, include_lhand and lhand, include_rhand and rhand)

def is_bad_sample(pose_sequence):
    # A missing body part -> bad sample
    for frame in pose_sequence:
        if any(is_body_part_missing(frame, include_face=False)):
            return True
    return False
    