import numpy as np

def translate_body(pose_sequence):
    #translates pose sequence so that middle between shoulders starts at the same position in the first frame
    center_between_shoulders = (pose_sequence[0][11] + pose_sequence[0][12])/2
    return pose_sequence - center_between_shoulders
