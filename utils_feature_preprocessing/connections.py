import numpy as np 

SUBSET_POSE_CONNECTIONS = frozenset([(11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

def connections_body(pose_sequence):                  
    #add vectors for each connection
    new_pose_seq = []
    for pose in pose_sequence:
        connection_sequence = []
        for connection in SUBSET_POSE_CONNECTIONS:
            connection_sequence.append(pose[connection[1]]-pose[connection[0]])
        new_pose_seq.append(connection_sequence)
    return np.array(new_pose_seq)