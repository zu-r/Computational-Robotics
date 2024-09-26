import numpy as np
import matplotlib.pyplot as plt


def interpolate_rigid_body(start_pose, goal_pose):
    x0, y0, theta0 = start_pose
    xN, yN, thetaN = goal_pose

    #find rotation
    direction = np.arctan2(yN-y0,xN-x0)
    

    #rotate
    steps = 10
    tpath = np.linspace(theta0, direction, steps)
    rpath = np.array([[x0,y0,theta] for theta in tpath])

    #go straight
    steps = 20
    xpath = np.linspace(x0,xN,steps)
    ypath = np.linspace(y0,yN,steps)
    tcolumn = np.full(steps,direction)
    spath = np.column_stack((xpath,ypath,tcolumn))

    #rotate to final orientation
    steps = 10
    tpath = np.linspace(direction, thetaN, steps)
    xcolumn = np.full(steps, xN)
    ycolumn = np.full(steps, yN)
    rpath2 = np.column_stack((xcolumn, ycolumn, tpath))


    path = np.vstack((rpath,spath,rpath2))
    
    print(path)
    return path.tolist()


def forward_propagate_rigid_body(start_pose, plan):
    path = np.array([start_pose])
    
    x, y, theta = start_pose
    
    for (vx, vy, vtheta), duration in plan:
        # print("velocity vector", vx * np.cos(theta), vy * np.sin(theta), vtheta)
        x_new = x + (vx * np.cos(theta) - vy * np.sin(theta)) * duration
        y_new = y + (vx * np.sin(theta) + vy * np.cos(theta)) * duration

        theta_new = theta + vtheta * duration
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        new_pose = np.array([x_new, y_new, theta_new])
        
        path = np.vstack([path, new_pose])
        
        x, y, theta = x_new, y_new, theta_new
    
    return path.tolist()



def visualize_path(path):

    fig, ax = plt.subplots()
    path = np.array(path)
    x_values = path[:, 0]
    y_values = path[:, 1]
    theta_values = path[:, 2]



    for i in range(len(path)):

        ax.cla()
        ax.plot(x_values, y_values, 'b-', label="Path")

        x = path[i][0]
        y = path[i][1]
        theta = path[i][2]

        x_arrow = [np.cos(theta), np.sin(theta)] 
        y_arrow = [-np.sin(theta), np.cos(theta)]

        robot_box = plt.Rectangle(
            (x - 0.25, y - 0.15),  
            0.5, 0.3,  
            angle=np.degrees(theta),  
            rotation_point = ('center'),
            fill=False, color='r'
        )
        ax.add_patch(robot_box)
        ax.quiver(x, y, x_arrow[0], x_arrow[1], angles='xy', scale_units='xy', scale=3, color='r', label="X-axis")
        ax.quiver(x, y, y_arrow[0], y_arrow[1], angles='xy', scale_units='xy', scale=3, color='g', label="Y-axis")

        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.pause(.1)
    plt.show()

def main():
    start_pose = (0, 0, 0)         
    goal_pose = (7.5, 0, np.pi / 2)  
    
    
    path = interpolate_rigid_body(start_pose, goal_pose)
    visualize_path(path)


    
    plan = [
        # Move forward with slight right rotation for 1.2 seconds
        ((1.0, 0.0, np.pi / 12), 2),  # Ends: slight right turn, x = 1.2
        
        # Move straight ahead for 0.7 seconds
        ((1.5, 0.0, 0.0), 1),  # Ends: straight, x = 2.25, y unchanged
        
        # Sharp right turn for 1 second
        ((1.0, 0.0, np.pi / 6), 1.0),  # Ends: larger right turn, x = 3.25
        
        # Move backward for 1.5 seconds, no rotation
        ((-1.0, 0.0, 0.0), 2),  # Ends: backward movement, x = 1.75
        
        # Move left while turning left for 0.8 seconds
        ((0.5, 0.5, -np.pi / 8), 1.5),  # Ends: move left and turn, x = 2.15, y = 0.4
       ]



    path = forward_propagate_rigid_body(start_pose,plan)
    visualize_path(path)

if __name__ == "__main__":
    main()