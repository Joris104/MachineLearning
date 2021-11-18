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
