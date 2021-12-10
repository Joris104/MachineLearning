import numpy as np

def get_average_of_bodyparts(pose_sequence):
    """
     Function that averages the keypoints over each body part
    """
    #body: 0->22
    #face:23->82
    #left hand:83->103
    #right hand:104->124
    #Take average per body part
    new_feature = []
    for frame in pose_sequence:
        body = frame[:23]
        face = frame[23:83]
        left_hand = frame[83:104]
        right_hand = frame[104:]
        new_feature.append([sum(body)/len(body),sum(face)/len(face),sum(left_hand)/len(left_hand),sum(right_hand)/len(right_hand)])
    
    #average over first halve frames & second halve frames
    if(len(pose_sequence)>1):
        N = len(pose_sequence)//2
        first_halve = new_feature[:N]
        second_halve = new_feature[N:]
        avg1_body = np.array([0.0, 0.0, 0.0])
        avg1_face = np.array([0.0, 0.0, 0.0])
        avg1_left_hand = np.array([0.0, 0.0, 0.0])
        avg1_right_hand = np.array([0.0, 0.0, 0.0])
        for row in first_halve:
            avg1_body += row[0]
            avg1_face += row[1]
            avg1_left_hand += row[2]
            avg1_right_hand += row[3]
        avg1_body /= len(first_halve)
        avg1_face /= len(first_halve) 
        avg1_left_hand /= len(first_halve)
        avg1_right_hand /= len(first_halve)
        avg2_body = np.array([0.0, 0.0, 0.0])
        avg2_face = np.array([0.0, 0.0, 0.0])
        avg2_left_hand = np.array([0.0, 0.0, 0.0])
        avg2_right_hand = np.array([0.0, 0.0, 0.0])
        for row in second_halve:
            avg2_body += row[0]
            avg2_face += row[1]
            avg2_left_hand += row[2]
            avg2_right_hand += row[3]
        avg2_body /= len(second_halve)
        avg2_face /= len(second_halve) 
        avg2_left_hand /= len(second_halve)
        avg2_right_hand /= len(second_halve)
        
        new_feature_avg = [avg1_body, avg2_body, avg1_face, avg2_face, avg1_left_hand, avg2_left_hand, avg1_right_hand, avg2_right_hand]
    else:
        #only one frame
        new_feature = new_feature[0]
        new_feature_avg = [new_feature[0], new_feature[0], new_feature[1], new_feature[1], new_feature[2], new_feature[2], new_feature[3], new_feature[3]]
    
    #flatten list of lists
    new_feature_avg = [coordinate for avg in new_feature_avg for coordinate in avg]
    return new_feature_avg