import numpy as np

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])

LHAND = 83
RHAND = 104


def vec_angle(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def hand_angles(pose_sequence):
    buf = np.zeros((pose_sequence.shape[0], 210 * 2))
    for k in range(buf.shape[0]):
        idx = 0
        for hand in [LHAND, RHAND]:
            vecs = [pose_sequence[k, hand + conn[0], :] - pose_sequence[k, hand + conn[1], :] for conn in HAND_CONNECTIONS]
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    buf[k, idx] = vec_angle(vecs[i], vecs[j])
                    idx += 1
    buf[np.isnan(buf)] = 0
    return buf
