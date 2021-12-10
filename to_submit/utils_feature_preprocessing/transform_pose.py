from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, norm
import numpy as np
import math

def move_hands_to_wrists(pose_sequence):
    pose_sequence_copy = pose_sequence.copy()
    for frame in pose_sequence_copy:
        # HAND_WRISTS = [83,104] BODY_WRISTS = [15,16]
        dist_vect_left = frame[83] - frame[15]
        dist_vect_right = frame[104] - frame[16]
        frame[83:104] -= dist_vect_left
        frame[104:125] -= dist_vect_right
    return pose_sequence_copy



def _translate_frame(frame, trans_vector):
        translated_frame = np.empty_like(frame)
        for i in range(len(frame)):
            translated_frame[i,:] = frame[i,:] + trans_vector
        return translated_frame

def _rotate_around(point, angle, axis):
    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(rotation_matrix(axis, angle), point)


def _rotate_frame(frame, angle, axis):
    rotated_frame = np.empty_like(frame)
    for i in range(len(frame)):
        rotated_frame[i,:] = _rotate_around(frame[i], angle, axis) 
    return rotated_frame


def _straighten_pose(frame):
    shoulder_avg = (frame[11]+frame[12])/2
    shoulder_dir = (frame[11]-frame[12])/np.linalg.norm(frame[11]-frame[12]) #length 1
    straight_shoulder_dir = [1,0,0]
    #print(shoulder_dir, frame[11], frame[12])
    
    # angle_between = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]
    angle_between = np.arccos(np.dot(shoulder_dir, straight_shoulder_dir))
    
    # Normal on shoulder-straight_shoulderplane
    normal_vector = np.cross(shoulder_dir, straight_shoulder_dir)
    
    # Rotate around average of shoulder according to angle_between around normal_vectore 
    new_frame = _translate_frame(frame, -shoulder_avg)
    new_frame = _rotate_frame(new_frame, angle_between, normal_vector)
    new_frame = _translate_frame(new_frame, shoulder_avg)
    
    return new_frame

def straighten_pose_sequence(pose_sequence):
    new_pose_sequence = []
    for frame in pose_sequence:
        new_pose_sequence.append(_straighten_pose(frame))
    return np.array(new_pose_sequence)

def _center_pose_around_shoulders(frame):
    shoulder_avg = (frame[11]+frame[12])/2
    return _translate_frame(frame, -shoulder_avg)
    
def center_pose_sequence_around_shoulders(pose_sequence):
    new_pose_sequence = []
    for frame in pose_sequence:
        new_pose_sequence.append(_center_pose_around_shoulders(frame))
    return np.array(new_pose_sequence)