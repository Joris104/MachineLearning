import numpy as np

def direction_handpalm(sequence):
    normals = []
    for frame in sequence:
        #left hand
        handpalm1 = frame[15]
        handpalm2 = frame[17]
        handpalm3 = frame[19]
        normal_left = np.cross(handpalm2 - handpalm1, handpalm3 - handpalm1)
        if not np.array_equal(normal_left, [0,0,0]):
            normal_left = normal_left/np.linalg.norm(normal_left)
        #right hand
        handpalm1 = frame[16]
        handpalm2 = frame[18]
        handpalm3 = frame[20]
        normal_right = np.cross(handpalm2 - handpalm1, handpalm3 - handpalm1)
        if not np.array_equal(normal_right, [0,0,0]):
            normal_right = normal_right/np.linalg.norm(normal_right)
        normals.append([normal_left,normal_right])
    normals = np.array(normals)
    return np.mean(normals, axis=0).flatten()