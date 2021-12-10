def correct_zeros_interpolation(pose_sequence, window=5):
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
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame < pose_sequence.shape[0]-1 and counter < window:
                            currentFrame += 1
                            counter += 1
                        pose_sequence[i][j][0] = pose_sequence[currentFrame][j][0]
                        pose_sequence[i][j][1] = pose_sequence[currentFrame][j][1]
                        pose_sequence[i][j][2] = pose_sequence[currentFrame][j][2]
                        
                    elif i == pose_sequence.shape[0]-1: #last frame
                        #search first non-zero value in the previous 5 frames
                        currentFrame = i-1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame > 0 and counter < window:
                            currentFrame -= 1
                            counter += 1
                        pose_sequence[i][j][0] = pose_sequence[currentFrame][j][0]
                        pose_sequence[i][j][1] = pose_sequence[currentFrame][j][1]
                        pose_sequence[i][j][2] = pose_sequence[currentFrame][j][2]
                        
                    else: #intermediate frame
                        
                        #search first non-zero value in the previous 5 frames
                        currentFrame = i-1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame > 0 and counter < window:
                            currentFrame -= 1
                            counter += 1
                        firstPrev = currentFrame
                        
                        #search first non-zero value in the next 5 frames
                        currentFrame = i+1
                        counter = 0
                        while pose_sequence[currentFrame][j][0] == 0 and currentFrame < pose_sequence.shape[0]-1 and counter < window:
                            currentFrame += 1
                            counter += 1
                        firstNext = currentFrame
                        
                        #take interpolation of both if neither is zero
                        if pose_sequence[firstPrev][j][0] != 0 and pose_sequence[firstNext][j][0] != 0:
                            k = i - firstPrev
                            n = firstNext - firstPrev
                            pose_sequence[i][j][0] = pose_sequence[firstPrev][j][0] + (pose_sequence[firstNext][j][0]-pose_sequence[firstPrev][j][0])*k/n
                            pose_sequence[i][j][1] = pose_sequence[firstPrev][j][1] + (pose_sequence[firstNext][j][1]-pose_sequence[firstPrev][j][1])*k/n
                            pose_sequence[i][j][2] = pose_sequence[firstPrev][j][2] + (pose_sequence[firstNext][j][2]-pose_sequence[firstPrev][j][2])*k/n
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