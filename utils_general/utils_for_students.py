import csv

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def create_submission_file(output_path, ids, predictions):
    """Write a submission file for Kaggle to the given path.
    
    :param output_path: The path to the output CSV file. Existing files will be overwritten.
    :param ids: The ids of the predictions.
    :predictions: A list of predictions.
    """
    with open(output_path, 'w', encoding='utf-8') as of:
        writer = csv.writer(of, lineterminator='\r')
        writer.writerow(['Id', 'Category'])
        for i, e in enumerate(ids):
            writer.writerow([e, predictions[i]])


def load_dataset_stage1(path, subset):
    """Loads an entire dataset of pre-extracted features. Only useful for stage 1.

    :param path: The dataset CSV file path.
    :param subset: Either 'train' or 'test'.
    :returns: A list of samples, where every sample is a dictionary containing a label, the signer ID and the feature list as a numpy array.
    """
    with open(path, 'r') as dataset_file:
        reader = csv.reader(dataset_file)

        samples = []
        if subset == 'train':
            _header = next(reader)  # Skip header
            for row in reader:
                label = int(row[0])
                signer = int(row[1])
                features = [float(e) for e in row[2:]]

                samples.append({
                    'label': label,
                    'signer': signer,
                    'features': np.array(features)
                })
        else:
            assert subset == 'test'
            for row in reader:
                # No label in test ;-)
                features = [float(e) for e in row]

                samples.append({
                    'features': np.array(features)
                })
        return samples


def load_dataset_stage2(path, subset):
    """Loads the samples from the given dataset. Only useful for stage 2.

    :param path: The dataset CSV file path.
    :param subset: Either 'train' or 'test'.
    :returns: A list of samples. For the test set, this is a dictionary containing the filename and the id.
    For the training set, it is a dictionary containing the filename, id, label and signer id.
    """
    samples = []
    with open(path, 'r') as label_file:
        reader = csv.reader(label_file)
        if subset == 'test':
            for row in reader:
                iD = int(row[0])
                filename = f'test_{iD:04d}.csv'  # 4 digits for id, add leading zeros if necessary
                samples.append({
                    'path': filename,
                    'id': iD
                })
        else:
            next(reader)  # Skip header
            for row in reader:
                iD, label, signer = row
                iD = int(iD)
                filename = f'train_{iD:04d}.csv'  # 4 digits for id, add leading zeros if necessary
                samples.append({
                    'path': filename,
                    'id': iD,
                    'label': label,
                    'signer': signer,
                })
    return samples


def load_sample_stage2(path):
    """Load a single sample. Only useful for stage 2.
    This only loads the features. You still need to extract the metadata from the provided metadata file(s).

    :param path: The sample CSV file path.
    :returns: A nested array of shape (L, 125, 3), where L is the number of frames.
    The number of keypoints is 125, and there are 3 coordinates per keypoint (x, y and z).
    """
    with open(path, 'r') as sample_file:
        reader = csv.reader(sample_file)
        header = next(reader)

        ix = header.index('x')
        iy = header.index('y')
        iz = header.index('z')
        iframe = header.index('frame')

        current_frame_index = 0
        sample_features = [[]]

        for row in reader:
            frame_index = int(row[iframe])
            if frame_index != current_frame_index:
                # New frame.
                assert frame_index == current_frame_index + 1
                current_frame_index += 1
                sample_features.append([])
            # After possibly creating new frame, we can append features.
            sample_features[current_frame_index].append([
                float(row[ix]),
                float(row[iy]),
                float(row[iz])
            ])
        for i in range(len(sample_features)):
            sample_features[i] = np.array(sample_features[i])

        return np.stack(sample_features)
    
# The below section contains code taken from the official MediaPipe source code repository.
# It was copied here to avoid requiring dependencies.

# The source files are:
#    - https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/hands_connections.py
#     - https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/pose_connections.py

# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])

# Copyright 2021 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe Pose connections."""

POSE_CONNECTIONS = frozenset([(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                              (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                              (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                              (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                              (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                              (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                              (29, 31), (30, 32), (27, 31), (28, 32)])

# Here is again our own code.

def visualize_pose(landmarks, ax):
    """Visualize a single pose.
    
    :param landmarks: A list of `(x, y, z)` tuples. 
    :param ax: A matplotlib axis.
    """
    # Plot the body landmarks in black.
    x = [l[0] for l in landmarks[:23]]
    y = [1-l[1] for l in landmarks[:23]]
    ax.scatter(x, y, color='black', s=0.8)
    for connection in POSE_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], color='black')
    
    # Plot the face landmarks in red.
    x = [l[0] for l in landmarks[23:83]]
    y = [1-l[1] for l in landmarks[23:83]]
    ax.scatter(x, y, color='red', s=0.5)
    
    # Plot the left hand landmarks in green.
    x = [l[0] for l in landmarks[83:104]]
    y = [1-l[1] for l in landmarks[83:104]]
    ax.scatter(x, y, color='green', s=0.8)
    for connection in HAND_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], color='green')
    
    # Plot the right hand landmarks in purple.
    x = [l[0] for l in landmarks[104:]]
    y = [1-l[1] for l in landmarks[104:]]
    ax.scatter(x, y, color='purple', s=0.8)
    for connection in HAND_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], color='purple')


def visualize_pose_3d(landmarks, ax=plt.axes(projection='3d')):
    """Visualize a single pose.
    
    :param landmarks: A list of `(x, y, z)` tuples. 
    :param ax: A matplotlib axis.
    """
    # Plot the body landmarks in black.
    x = [l[0] for l in landmarks[:23]]
    y = [1-l[1] for l in landmarks[:23]]
    z = [l[2] for l in landmarks[:23]]
    ax.scatter3D(x, y, z, color='black', s=0.8)
    for connection in POSE_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot3D([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], [z[connection[0]], z[connection[1]]], color='black')

    # Plot the face landmarks in red.
    x = [l[0] for l in landmarks[23:83]]
    y = [1-l[1] for l in landmarks[23:83]]
    z = [l[2] for l in landmarks[23:83]]
    ax.scatter3D(x, y, z, color='red', s=0.5)

    # Plot the left hand landmarks in green.
    x = [l[0] for l in landmarks[83:104]]
    y = [1-l[1] for l in landmarks[83:104]]
    z = [l[2] for l in landmarks[83:104]]
    ax.scatter3D(x, y, z, color='green', s=0.8)
    for connection in HAND_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot3D([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], [z[connection[0]], z[connection[1]]], color='green')

    # Plot the right hand landmarks in purple.
    x = [l[0] for l in landmarks[104:]]
    y = [1-l[1] for l in landmarks[104:]]
    z = [l[2] for l in landmarks[104:]]
    ax.scatter3D(x, y, z, color='purple', s=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for connection in HAND_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot3D([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], [z[connection[0]], z[connection[1]]], color='purple') 


def label_encoder():
    """Encodes the labels as integers.
    You can use this to go from class labels to integers (`transform`) or from
    integers back to class labels (`inverse_transform`).
    
    :returns: The `LabelEncoder` instance."""

    CLASSES = ['c.AF', 'c.OOK', 'ZELFDE-A', 'AUTO-RIJDEN-A', 'HEBBEN-A', 'HAAS-oor', 'AANKOMEN-A',
           'SCHILDPAD-Bhanden', 'WAT-A', 'c.ZIEN', 'NAAR-A', 'MOETEN-A', 'C: 1', 'GOED-A', 'C: 2']

    encoder = LabelEncoder()
    encoder.fit(CLASSES)
    return encoder
