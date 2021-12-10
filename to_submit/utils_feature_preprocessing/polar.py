import numpy as np


def to_polar_coordinates(pose_sequence):
    """
    Function for transforming to polar coordinates
    """
    def transform_coordinate(coordinate):
        polar = []
        polar.append(np.sqrt(pow(coordinate[0], 2) + pow(coordinate[1], 2) + pow(coordinate[2], 2)))
        if coordinate[0] == 0 or coordinate[1] == 0:
            polar.append(0)
        else:
            polar.append(np.arccos(coordinate[0]/np.sqrt(pow(coordinate[0], 2) + pow(coordinate[1], 2))) * np.sign(coordinate[1]))
        if polar[0] == 0:
            polar.append(0)
        else:
            polar.append(np.arccos(coordinate[2]/polar[0]))
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