import numpy as np

def extract_features(pose_sequence):
    """
     Function that removes face keypoints
    """
    #body: 0->22
    #face:23->82
    #left hand:83->103
    #right hand:104->124
    #Take average per body part
    new_feature = []
    for frame in pose_sequence:
        body = frame[:23].flatten()
        left_hand = frame[83:104].flatten()
        right_hand = frame[104:].flatten()
        new_feature.append(np.concatenate((body,left_hand,right_hand)))
    
    #average over first halve frames & second halve frames
    if(len(pose_sequence)>1):
        N = len(pose_sequence)//2
        
        first_halve = new_feature[:N]
        first_halve = np.array(first_halve)
        first_halve = sum(first_halve)/len(first_halve)
        
        second_halve = new_feature[N:]
        second_halve = np.array(second_halve)
        second_halve = sum(second_halve)/len(second_halve)
        
        new_feature = np.concatenate((first_halve,second_halve))
        
    else:
        #only one frame
        new_feature = np.concatenate((new_feature[0],new_feature[0]))
    
    #flatten list of lists
    return new_feature