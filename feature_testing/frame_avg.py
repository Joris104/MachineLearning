from utils_feature_preprocessing import split_features_into_frames

def extract_features(pose_sequence):
    # You should implement this function to return better features!
    
    pose_sequence = split_features_into_frames(pose_sequence, k=2)
    
    return pose_sequence