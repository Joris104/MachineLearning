import numpy as np

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

LHAND = 83
RHAND = 104


def vec_angle(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def finger_angles(buf, idx, pose_sequence, k, conns, hand):
    vecs = [pose_sequence[k,hand+conn[1],:] - pose_sequence[k,hand+conn[0],:] for conn in conns]
    buf[k,idx+0] = vec_angle(vecs[0], vecs[1])
    buf[k,idx+1] = vec_angle(vecs[1], vecs[2])

def hand_angles(pose_sequence):
    buf = np.zeros((pose_sequence.shape[0], 10 * 2))
    for k in range(buf.shape[0]):
        finger_angles(buf, 0, pose_sequence, k, HAND_THUMB_CONNECTIONS, LHAND)
        finger_angles(buf, 2, pose_sequence, k, HAND_INDEX_FINGER_CONNECTIONS, LHAND)
        finger_angles(buf, 4, pose_sequence, k, HAND_MIDDLE_FINGER_CONNECTIONS, LHAND)
        finger_angles(buf, 6, pose_sequence, k, HAND_RING_FINGER_CONNECTIONS, LHAND)
        finger_angles(buf, 8, pose_sequence, k, HAND_PINKY_FINGER_CONNECTIONS, LHAND)
        finger_angles(buf, 10, pose_sequence, k, HAND_THUMB_CONNECTIONS, RHAND)
        finger_angles(buf, 12, pose_sequence, k, HAND_INDEX_FINGER_CONNECTIONS, RHAND)
        finger_angles(buf, 14, pose_sequence, k, HAND_MIDDLE_FINGER_CONNECTIONS, RHAND)
        finger_angles(buf, 16, pose_sequence, k, HAND_RING_FINGER_CONNECTIONS, RHAND)
        finger_angles(buf, 18, pose_sequence, k, HAND_PINKY_FINGER_CONNECTIONS, RHAND)
    buf[np.isnan(buf)] = 0
    return buf
