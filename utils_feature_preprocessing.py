import numpy as np

def split_features_into_frames(pose_sequence, k=2):
    if k < 1 or not pose_sequence.shape[0]:
        raise AttributeError

    n = pose_sequence.shape[0]

    # Sequence indices indicate the start- and endframe that will be combined into a new aggregated frame
    # e.g. n=6, k=4 -> [0, 1, 3, 4, 6]
    # e.g. n=4, k=6 -> [0, 0, 1, 2, 2, 3, 3]
    # Note: if e.g. ..,3,3,...this becomes ..,3,4,.. (1 frame)
    sequence_indices = [0]*(k+1)
    sequence_increment = n/k
    sequence_counter = 0
    for i in range(k+1):
        sequence_indices[i] = int(np.floor(sequence_counter))
        sequence_counter += sequence_increment
    #print(sequence_indices)

    # Make aggregated frame according to indices
    new_pose_sequence = []
    for i in range(k):
        slice_start = sequence_indices[i]
        slice_end = sequence_indices[i+1]
        if slice_start == slice_end:
            slice_end += 1
        frame = np.mean(pose_sequence[slice_start:slice_end], axis=0)
        new_pose_sequence.append(frame)
    new_pose_sequence = np.stack(new_pose_sequence).reshape(-1)
    new_pose_sequence = np.nan_to_num(new_pose_sequence)

    return new_pose_sequence

def correct_zeros(pose_sequence):
    """
     Function that corrects zero values
    """
    if pose_sequence.shape[0]>1:
        for i in range(pose_sequence.shape[0]):
            for j in range(pose_sequence.shape[1]):
                
                if pose_sequence[i][j][0] == 0 and pose_sequence[i][j][1] == 0 and pose_sequence[i][j][2] == 0:
                    
                    if i == 0: #first frame
                        #search first non-zero value in the next 5 frames
                        currentFrame = i+1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame < pose_sequence.shape[0]-1 and counter < 5:
                            currentFrame += 1
                            counter += 1
                        pose_sequence[i][j][0] = pose_sequence[currentFrame][j][0]
                        pose_sequence[i][j][1] = pose_sequence[currentFrame][j][1]
                        pose_sequence[i][j][2] = pose_sequence[currentFrame][j][2]
                        
                    elif i == pose_sequence.shape[0]-1: #last frame
                        #search first non-zero value in the previous 5 frames
                        currentFrame = i-1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame > 0 and counter < 5:
                            currentFrame -= 1
                            counter += 1
                        pose_sequence[i][j][0] = pose_sequence[currentFrame][j][0]
                        pose_sequence[i][j][1] = pose_sequence[currentFrame][j][1]
                        pose_sequence[i][j][2] = pose_sequence[currentFrame][j][2]
                        
                    else: #intermediate frame
                        
                        #search first non-zero value in the previous 5 frames
                        currentFrame = i-1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame > 0 and counter < 5:
                            currentFrame -= 1
                            counter += 1
                        firstPrev = currentFrame
                        
                        #search first non-zero value in the next 5 frames
                        currentFrame = i+1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame < pose_sequence.shape[0]-1 and counter < 5:
                            currentFrame += 1
                            counter += 1
                        firstNext = currentFrame
                        
                        #take average of both if neither is zero
                        if pose_sequence[firstPrev][j][0] != 0 and pose_sequence[firstNext][j][0] != 0:
                            pose_sequence[i][j][0] = (pose_sequence[firstPrev][j][0]+pose_sequence[firstNext][j][0])/2
                            pose_sequence[i][j][1] = (pose_sequence[firstPrev][j][1]+pose_sequence[firstNext][j][1])/2
                            pose_sequence[i][j][2] = (pose_sequence[firstPrev][j][2]+pose_sequence[firstNext][j][2])/2
                        #take non-zero if one is zero
                        elif pose_sequence[firstPrev][j][0] == 0 and pose_sequence[firstNext][j][0] != 0:
                            pose_sequence[i][j][0] = pose_sequence[firstNext][j][0]
                            pose_sequence[i][j][1] = pose_sequence[firstNext][j][1]
                            pose_sequence[i][j][2] = pose_sequence[firstNext][j][2]
                        elif pose_sequence[firstPrev][j][0] != 0 and pose_sequence[firstNext][j][0] == 0:
                            pose_sequence[i][j][0] = pose_sequence[firstPrev][j][0]
                            pose_sequence[i][j][1] = pose_sequence[firstPrev][j][1]
                            pose_sequence[i][j][2] = pose_sequence[firstPrev][j][2]

    return pose_sequence