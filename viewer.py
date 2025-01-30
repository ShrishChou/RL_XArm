import pybullet as p
import pybullet_data
import time
import numpy as np
import math
# Connect to PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane
planeId = p.loadURDF("plane.urdf")

# Load xArm7 robot
robot_urdf_path = "/output.urdf"  # Make sure this path is correct
robot_start_pos = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_urdf_path,
                   basePosition=robot_start_pos,
                   baseOrientation=robot_start_orientation,
                   useFixedBase=True)  # Fix the base of the robot

# Load object
object1_id = p.loadURDF("/housing.urdf", basePosition=[0.75, 0, 0], baseOrientation=[0, 0, 0, 0.1])
object2_id = p.loadURDF("/housing.urdf", basePosition=[0.4, 0, 0])
object3_id = p.loadURDF("/housing.urdf", basePosition=[0.5, 0.2, 0])

# Get the number of joints
num_joints = p.getNumJoints(robot_id)

# Initialize joint positions
joint_positions = [0] * num_joints

# Define multipliers for joint relationships
JOINT_10_MULTIPLIER = 1  # Joint 10 will move twice as much as joint 9
JOINT_13_MULTIPLIER = 1  # Joint 13 will move 1.5 times as much as joint 12

# Control parameters
joint_index = 0  # Start with the first joint
position_step = 0.05  # Change in position per key press
velocity_step = 0.1  # Change in velocity per key press

print("\nControl Instructions:")
print("Arrow Up/Down: Change joint position")
print("Arrow Left/Right: Change joint velocity")
print("N: Next joint")
print("P: Previous joint")
print(f"\nJoint Relationships:")
print(f"Joint 10 moves {JOINT_10_MULTIPLIER}x the amount of joint 9")
print(f"Joint 12 moves the same amount as joint 9")
print(f"Joint 13 moves {JOINT_13_MULTIPLIER}x the amount of joint 12")

# Main loop
while True:
    link_state = p.getLinkState(robot_id, 6,computeForwardKinematics=True)
    position, orientation = link_state[4], link_state[5]
    
    # Get the rotation matrix of the end effector
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    
    forward_vector = rotation_matrix[:, 2]  # First column of the rotation matrix

    # Define the vertical axis (Z-axis in PyBullet)
    vertical_axis = np.array([0, 0, 1])
    dot_product = np.dot(forward_vector, vertical_axis)

    # Calculate the angle between the forward vector and the vertical axis
    angle_from_vertical = np.arccos(dot_product)  # Result in radians
    angle_from_vertical_deg = abs(180-np.degrees(angle_from_vertical)) 
    if forward_vector[0]<0:
        # Calculate the dot product between the forward vector and the vertical axis
         # Convert to degrees
        angle_from_vertical_deg=180+180-angle_from_vertical_deg
    
    # dot_product = np.dot(forward_vector, vertical_axis)
    # angle_from_vertical = np.arccos(dot_product)
    # angle_from_vertical_deg = abs(180-np.degrees(angle_from_vertical))

    object1_position, _ = p.getBasePositionAndOrientation(object1_id)
    object1_position = np.array(object1_position)
    object1_position[2] += 0.23
    horizontal_dist = np.linalg.norm(position - object1_position)
    
    print(f"Angle from vertical: {angle_from_vertical_deg} degrees"," ",horizontal_dist)
    keys = p.getKeyboardEvents()
    for key, value in keys.items():
        if value & p.KEY_WAS_TRIGGERED:
            if key == p.B3G_UP_ARROW:  # Increase joint position
                joint_positions[joint_index] += position_step
                # Update related joints
                if joint_index == 9:
                    joint_positions[10] = joint_positions[9] * JOINT_10_MULTIPLIER
                    joint_positions[11] = joint_positions[9]
                    joint_positions[12] = joint_positions[9]  # Same movement as joint 9
                    joint_positions[13] = joint_positions[12] * JOINT_13_MULTIPLIER
                    joint_positions[14] = joint_positions[12]
            elif key == p.B3G_DOWN_ARROW:  # Decrease joint position
                joint_positions[joint_index] -= position_step
                # Update related joints
                if joint_index == 9:
                    joint_positions[10] = joint_positions[9] * JOINT_10_MULTIPLIER
                    joint_positions[11] = joint_positions[9]
                    joint_positions[12] = joint_positions[9]  # Same movement as joint 9
                    joint_positions[13] = joint_positions[12] * JOINT_13_MULTIPLIER
                    joint_positions[14] = joint_positions[12]
            elif key == p.B3G_RIGHT_ARROW:  # Increase joint velocity
                p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL, targetVelocity=velocity_step)
            elif key == p.B3G_LEFT_ARROW:  # Decrease joint velocity
                p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL, targetVelocity=-velocity_step)
            elif key == ord('n'):  # Switch to next joint
                joint_index = (joint_index + 1) % num_joints
            elif key == ord('p'):  # Switch to previous joint
                joint_index = (joint_index - 1) % num_joints
            
    # Apply joint positions
    for i in range(num_joints):
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_positions[i],
            force=500  # Maximum force to apply
        )
    
    # Display current joint info
    print(f"\rActive Joint: {joint_index} | Position: {joint_positions[joint_index]:.2f}", end="")
    
    # Step the simulation
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()