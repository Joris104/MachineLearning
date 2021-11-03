import math
import numpy as np

def extract_features(pose_sequence, m=3):
    
    n = pose_sequence.shape[1]
    data = np.zeros((sum(range(m+1)), n, pose_sequence.shape[2]))
    
    for i in range(m):
        layer = i + 1
        num_avgs = layer
        offset = sum(range(layer))
        
        for j in range(num_avgs):
            head = j * pose_sequence.shape[0] / num_avgs
            tail = (j + 1) * pose_sequence.shape[0] / num_avgs
            
            head_idx = math.floor(head)
            tail_idx = math.ceil(tail) - 1
            
            for k in range(n):
                if head_idx != tail_idx:
                    data[j+offset,k] += pose_sequence[head_idx,k] * (1 - (head - head_idx))
                    for l in range(head_idx + 1, tail_idx):
                        data[j+offset,k] += pose_sequence[l,k]
                    data[j+offset,k] += pose_sequence[tail_idx,k] * (tail - tail_idx)
                    data[j+offset,k] /= tail - head
                else:
                    data[j+offset,k] += pose_sequence[head_idx,k]

    return data.reshape(sum(range(m+1)) * n * pose_sequence.shape[2])
