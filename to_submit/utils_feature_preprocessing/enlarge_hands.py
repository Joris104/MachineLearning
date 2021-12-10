import numpy as np 

def enlarge_hands(pose_sequence, scale_factor=1.5):
    # Enlarge each hand
    def enlarge_hand(pose_sequence, scale_factor, start_index,end_index):
        for frame in range(pose_sequence.shape[0]):
            avg_hand = np.mean(pose_sequence[frame][start_index:end_index], axis=0)
            for point_index in range(start_index,end_index):
                point = pose_sequence[frame][point_index]
                dist = point - avg_hand
                dist *= scale_factor
                pose_sequence[frame][point_index] = avg_hand + dist  
        return pose_sequence
                
    pose_sequence = enlarge_hand(pose_sequence, scale_factor, 83, 104)
    pose_sequence = enlarge_hand(pose_sequence, scale_factor, 104, 125)
    return pose_sequence
