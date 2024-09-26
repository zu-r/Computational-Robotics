import numpy as np
import matplotlib.pyplot as plt
from component_1 import check_SOn, check_quaternion

def random_rotation_matrix(naive: bool):
    if naive:
        roll = np.random.uniform(0,2*np.pi)
        pitch = np.random.uniform(0,2*np.pi)
        yaw = np.random.uniform(0,2*np.pi)

        X = np.array(

            [[1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]]

        )

        Y = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]]
        )
        
        Z = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]]
        )

        print(roll,pitch,yaw, "big gap", X,Y,Z, "big gap", Z @ Y @ X,"big gap",check_SOn(Z @ Y @ X, 0.01))
        return Z @ Y @ X
    else:
        u1, u2, u3 = np.random.uniform(0, 1, 3)

        #Algorithm 2
        q = np.array(
            [np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
             np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
             np.sqrt(u1) * np.sin(2 * np.pi * u3),
             np.sqrt(u1) * np.cos(2 * np.pi * u3)]
            )
        w, x, y, z = q

        r00 = 1 - 2*y**2 - 2*z**2
        r01 = 2*x*y - 2*z*w
        r02 = 2*x*z + 2*y*w
            
        # Second row of the rotation matrix
        r10 = 2*x*y + 2*z*w
        r11 = 1 - 2*x**2 - 2*z**2
        r12 = 2*y*z - 2*x*w
            
        # Third row of the rotation matrix
        r20 = 2*x*z - 2*y*w
        r21 = 2*y*z + 2*x*w
        r22 = 1 - 2*x**2 - 2*y**2
            
        print(check_SOn(np.array([[r00, r01, r02],[r10, r11, r12],[r20, r21, r22]]),.01))    
        # 3x3 rotation matrix
        return np.array(
                    [[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]]
                )

        

def random_quaternion(naive: bool):
    if naive:
        roll = np.random.uniform(0, 2*np.pi)
        pitch = np.random.uniform(0, 2*np.pi)
        yaw = np.random.uniform(0, 2*np.pi)

        cosr, sinr = np.cos(roll/2), np.sin(roll/2)
        cosp, sinp = np.cos(pitch/2), np.sin(pitch/2)
        cosy, siny = np.cos(yaw/2), np.sin(yaw/2)

        w = (cosr * cosp * cosy) + (sinr * sinp * siny)
        x = (sinr * cosp * cosy) - (cosr * sinp * siny)
        y = (cosr * sinp * cosy) + (sinr * cosp * siny)
        z = (cosr * cosp * siny) - (sinr * sinp * cosy)
        

        return [w,x,y,z]

    else:
        x1, x2, x3 = np.random.uniform(0,1,3)

        theta = 2 * np.pi * x1
        phi = 2 * np.pi * x2

        r = np.sqrt(1-x3)

        V = np.array(
            [r * np.cos(phi),
             r * np.sin(phi),
             np.sqrt(x3)]
        )

        w = np.cos(theta/2)
        x = V[0] * np.sin(theta/2)
        y = V[1] * np.sin(theta/2)
        z = V[2] * np.sin(theta/2)

        return [w,x,y,z]
    

def apply_rotation(R, v):
    return R @ v

# Visualizing rotation samples
def visualize_random_rotations_comparison(num_samples=1000):
    points_naive = []
    points_uniform = []
    
    for i in range(num_samples):
        # Naive method: using random Euler angles
        R_naive = random_rotation_matrix(naive=True)
        rotated_point_naive = apply_rotation(R_naive, np.array([0, 0, 1]))
        points_naive.append(rotated_point_naive)
        
        # Uniform method: using quaternions
        R_uniform = random_rotation_matrix(naive=False)
        rotated_point_uniform = apply_rotation(R_uniform, np.array([0, 0, 1]))
        points_uniform.append(rotated_point_uniform)

    points_naive = np.array(points_naive)
    points_uniform = np.array(points_uniform)

    # Plotting the results
    fig = plt.figure(figsize=(12, 6))

    # Naive sampling subplot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points_naive[:, 0], points_naive[:, 1], points_naive[:, 2], c='r', marker='o', s=1)
    ax1.set_title('Naive Sampling')
    
    # Uniform sampling subplot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points_uniform[:, 0], points_uniform[:, 1], points_uniform[:, 2], c='b', marker='o', s=1)
    ax2.set_title('Uniform Sampling')

    # Add sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax1.plot_surface(x, y, z, color='c', alpha=0.1)
    ax2.plot_surface(x, y, z, color='c', alpha=0.1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

# Example usage
visualize_random_rotations_comparison(num_samples=1000)