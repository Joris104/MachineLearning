import numpy as np

def get_direction_unit(sequence, kp1, kp2):
    l = []
    for frame in sequence:
        bottom = frame[kp1][0:2]
        top = frame[kp2][0:2]
        direction = bottom-top
        if(np.linalg.norm(direction)>0):
            direction = direction/np.linalg.norm(direction)
        l.append(direction)
    return np.mean(l, axis=0)

def amount_of_stretched_fingers(sequence):
    
    left_wrist_to_bottom_index = get_direction_unit(sequence, 83, 88)
    left_wrist_to_top_index = get_direction_unit(sequence, 83, 91)
    left_wrist_to_bottom_middle = get_direction_unit(sequence, 83, 92)
    left_wrist_to_bottom_middle = get_direction_unit(sequence, 83, 95)
    
    right_wrist_to_bottom_index = get_direction_unit(sequence, 104, 109)
    right_wrist_to_top_index = get_direction_unit(sequence, 104, 112)
    right_wrist_to_bottom_middle = get_direction_unit(sequence, 104, 113)
    right_wrist_to_bottom_middle = get_direction_unit(sequence, 104, 116)
    
    #difference
    diff_index_left = np.linalg.norm(left_wrist_to_bottom_index - left_wrist_to_top_index)
    diff_middle_left = np.linalg.norm(left_wrist_to_bottom_middle - left_wrist_to_bottom_middle)    
    diff_index_right = np.linalg.norm(right_wrist_to_bottom_index - right_wrist_to_top_index)
    diff_middle_right = np.linalg.norm(right_wrist_to_bottom_middle - right_wrist_to_bottom_middle)

    index_threshold = 0.15
    middle_threshold = 0.15
    #left
    amount_left = 0
    if diff_index_left < index_threshold and diff_index_left > 0:
        amount_left += 1
        if diff_middle_left < middle_threshold:
            amount_left += 1
    
    #right
    amount_right = 0
    if diff_index_right < index_threshold and diff_index_right > 0:
        amount_right += 1
        if diff_middle_right < middle_threshold:
            amount_right += 1
    
    return np.array([np.max([amount_left, amount_right])])