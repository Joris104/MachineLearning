import numpy as np


def select_frames(sequence, start, end):
    if (start < 0 and end < 0):
        #notice that in this case start = -1 means till last element, therefore the plus one
        if (sequence.shape[0] >= -end):
            selected = sequence[sequence.shape[0]+end:sequence.shape[0]+start+1]
        elif (sequence.shape[0] > -start):
            selected = sequence[:sequence.shape[0]+start+1]
        else:
            selected = sequence
    elif (start >= 0 and end >= 0):
        if (sequence.shape[0] >= end):
            selected = sequence[start:end]
        elif (sequence.shape[0] > start):
            selected = sequence[start:]
        else:
            selected = sequence
    else:
        raise ValueError("inconsistent indexing")

    return transform_to_k_frames(selected, k=1)

def transform_to_k_frames(pose_sequence, k=2):
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
    new_pose_sequence = np.nan_to_num(new_pose_sequence)

    return new_pose_sequence

def frames_to_features(pose_sequence):
    return np.stack(pose_sequence).reshape(-1)