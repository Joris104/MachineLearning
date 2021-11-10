def extract_features(pose_sequence):
    """
     Function that corrects zero values
    """
    if pose_sequence.shape[0]>1:
        for i in range(pose_sequence.shape[0]):
            for j in range(pose_sequence.shape[1]):
                if pose_sequence[i][j][0] == 0 and pose_sequence[i][j][1] == 0 and pose_sequence[i][j][2] == 0:
                    if i == 0:
                        pose_sequence[i][j][0] = pose_sequence[i+1][j][0]
                        pose_sequence[i][j][1] = pose_sequence[i+1][j][1]
                        pose_sequence[i][j][2] = pose_sequence[i+1][j][2]
                    elif i == pose_sequence.shape[0]-1:
                        pose_sequence[i][j][0] = pose_sequence[i-1][j][0]
                        pose_sequence[i][j][1] = pose_sequence[i-1][j][1]
                        pose_sequence[i][j][2] = pose_sequence[i-1][j][2]
                    else:
                        pose_sequence[i][j][0] = (pose_sequence[i-1][j][0]+pose_sequence[i+1][j][0])/2
                        pose_sequence[i][j][1] = (pose_sequence[i-1][j][1]+pose_sequence[i+1][j][1])/2
                        pose_sequence[i][j][2] = (pose_sequence[i-1][j][2]+pose_sequence[i+1][j][2])/2
    return pose_sequence