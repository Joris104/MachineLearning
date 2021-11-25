import numpy as np

def minimumFingerEye(sequence):
    #find minimum distance between left eye and left index finger or right eye and right index finger
    minimum = 9999
    for frame in sequence:
        #distance left eye <-> left index finger
        leftEye = frame[2]
        leftIndexFinger = frame[91]
        d = np.linalg.norm(leftEye - leftIndexFinger)
        if d < minimum:
            minimum = d
        #distance right eye <-> right index finger
        rightEye = frame[5]
        rightIndexFinger = frame[112]
        d = np.linalg.norm(rightEye - rightIndexFinger)
        if d < minimum:
            minimum = d
    return minimum