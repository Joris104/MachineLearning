import math
import numpy as np

def extract_features(pose_sequence):
    data = np.var(pose_sequence, axis=0)
    return data.reshape(pose_sequence.shape[1] * pose_sequence.shape[2])
