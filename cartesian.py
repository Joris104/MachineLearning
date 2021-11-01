import numpy as np


def extract_features(pose_sequence):
    """
     Function for checking the actual added value of transforming to polar coordinates
    """
    def transform_coordinate(coordinate):
        polar = [coordinate[0], coordinate[1], coordinate[2]]
        return polar
    
    features = []
    if len(pose_sequence) > 1:
        #construct 2 poses from averaging first and second half
        N = int(len(pose_sequence)/2)
        np_pose_sequence = np.array(pose_sequence)
        first_half = np_pose_sequence[0]
        for idx_pose in range(1, N):
            first_half += np_pose_sequence[idx_pose]

        first_half /= N

        second_half = np_pose_sequence[N]
        for idx_pose in range(N+1, len(pose_sequence)):
            second_half += np_pose_sequence[idx_pose]

        second_half /= len(pose_sequence)-N
        
        #transform coordinates
        for half in [first_half, second_half]:
            for coordinate in half:
                polar = transform_coordinate(coordinate)
                features.extend(polar)
    else:
        #if one frame add 2 times
        for i in range(0, 2):
            for coordinate in pose_sequence[0]:
                polar = transform_coordinate(coordinate)
                features.extend(polar)
                
    return features