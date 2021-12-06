from mpl_toolkits import mplot3d

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
    for connection in HAND_CONNECTIONS:
        if connection[0] < len(x) and connection[1] < len(x):
            ax.plot3D([x[connection[0]], x[connection[1]]], [y[connection[0]], y[connection[1]]], [z[connection[0]], z[connection[1]]], color='purple')