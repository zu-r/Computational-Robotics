import numpy as np
import matplotlib.pyplot as plt

def interpolate_arm(start, goal):
    steps = 10
    path = []
    for i in range(steps):
        t = i / (steps - 1)
        theta0 = start[0] * (1 - t) + goal[0] * t
        theta1 = start[1] * (1 - t) + goal[1] * t
        path.append([theta0, theta1])
    print(path)
    return path

def forward_propagate_arm(start_pose, plan):
    path = np.array([start_pose])
    theta0, theta1 = start_pose

    for (v0, v1), time in plan:
        t0 = theta0 + v0 * time
        t1 = theta1 + v1 * time

        pose = np.array([t0,t1])
        path = np.vstack([path,pose])

        theta0,theta1 = t0,t1

    return path.tolist()

def visualize_arm_path(path):
    fig, ax = plt.subplots()

    L1 = 2  
    L2 = 1.5

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    end_effector_path = []
    for i in range(len(path)):
        theta1 = path[i][0]
        theta2 = path[i][1]
        
        # Calculate joint positions
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)
        
        end_effector_path.append([x2, y2])
    
    # Extract the full path coordinates
    path_x = [pos[0] for pos in end_effector_path]
    path_y = [pos[1] for pos in end_effector_path]

    # Plot the complete end effector path before animation
    ax.plot(path_x, path_y, 'g--', label="End Effector Path", alpha=0.6)
    

    for i in range(len(path)):
        
        ax.cla()
        ax.plot(path_x, path_y, 'g--', label="End Effector Path", alpha=0.6)

        theta1 = path[i][0] 
        theta2 = path[i][1] 


        x0, y0 = 0, 0  
        x1 = L1 * np.cos(theta1)
        y1 = L1 * np.sin(theta1)  
        x2 = x1 + L2 * np.cos(theta1 + theta2)
        y2 = y1 + L2 * np.sin(theta1 + theta2)  


        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_aspect('equal')

        ax.plot([x0, x1], [y0, y1], 'b-o', label="L1")  
        ax.plot([x1, x2], [y1, y2], 'r-o', label="L2")  

        
        plt.pause(0.3) 
        
    plt.show()


def run_test_cases():
    paths = []


    start1 = [0, 0]
    goal1 = [np.pi / 4, np.pi / 4]
    path1 = interpolate_arm(start1, goal1)
    paths.append(path1)
    print("Test Case 1 Output:", path1)


    start2 = [0, 0]
    goal2 = [2 * np.pi, 2 * np.pi]
    path2 = interpolate_arm(start2, goal2)
    paths.append(path2)
    print("Test Case 2 Output:", path2)


    start3 = [np.pi, np.pi]
    goal3 = [0, 0]
    path3 = interpolate_arm(start3, goal3)
    paths.append(path3)
    print("Test Case 3 Output:", path3)


    start4 = [0, 0]
    plan4 = [((0, 0.1), 5)]
    path4 = forward_propagate_arm(start4, plan4)
    paths.append(path4)
    print("Test Case 4 Output:", path4)


    start5 = [0, 0]
    plan5 = [
        ((0.1, 0), 5),     
        ((0, 0.2), 5),     
        ((-0.1, -0.2), 5)  
    ]
    path5 = forward_propagate_arm(start5, plan5)
    paths.append(path5)
    print("Test Case 5 Output:", path5)


    for idx, path in enumerate(paths):
        print(f"Visualizing Path for Test Case {idx + 1}")
        visualize_arm_path(path)

if __name__ == "__main__":
    run_test_cases()
