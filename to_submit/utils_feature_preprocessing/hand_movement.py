import numpy as np

def hand_movement(keypoints):
    lhand = keypoints[:,83:104,:]
    rhand = keypoints[:,104:,:]
    #print(rhand)
    # for kp in keypoints[1:]:
        # prev_lhand = lhand
        # prev_rhand = rhand
        # lhand = kp[83:104,:]
        # rhand = kp[104:,:]
        # lhand_mov = np.abs(lhand-prev_lhand)+lhand_mov
        # rhand_mov = np.abs(rhand-prev_rhand)+rhand_mov
    
    lhand_mov = np.abs(np.diff(lhand, axis=0))
    rhand_mov = np.abs(np.diff(rhand, axis=0))
    lhand_mov = np.sum(lhand_mov, axis=0)
    lhand_mov = np.sum(lhand_mov, axis=0)
    rhand_mov = np.sum(rhand_mov, axis=0)
    rhand_mov = np.sum(rhand_mov, axis=0)

    hand_mov = lhand_mov + rhand_mov
    return np.concatenate((hand_mov, lhand_mov, rhand_mov))