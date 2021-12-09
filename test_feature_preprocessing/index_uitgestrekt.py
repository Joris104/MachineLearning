import numpy as np

def index_uitgestrekt(poses):
    thumb = np.array([0,1, 2, 3, 4])
    index = np.array([0,5, 6, 7, 8])
    middle = np.array([0, 9, 10, 11, 12])
    ring = np.array([0,13, 14, 15, 16])
    pinky = np.array([0,17, 18, 19, 20])
    
    fingers = np.vstack((thumb, index, middle, ring, pinky))
    features = np.empty(len(poses), dtype=np.object)
    for i,frame in enumerate(poses):
        f = np.zeros((5,2))
        for idx,finger in enumerate(fingers) :
            left =  frame[83+finger,:]
            right = frame[104+finger,:]
            #print(finger)
            #print(frame)
            lcorr = np.corrcoef(left[:,0],left[:,1])[0,1]
            rcorr = np.corrcoef(right[:,0],right[:,1])[0,1]
            
            f[idx] = [lcorr,rcorr]
        features[i] = np.array(f)
    mean = np.mean(features, axis=0)
    
    return np.nan_to_num(mean).flatten()