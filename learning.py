import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
from gymnasium import spaces
from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
def get_observation(robot_id, object1_id):
    # Get robot joint states (joints 0-6 for motion, 9-14 for claw)
    joint_states = p.getJointStates(robot_id, range(15))  # All 15 joints
    joint_positions = [state[0] for i, state in enumerate(joint_states) if i < 7 or i >= 9]  # Exclude joints 7 and 8
    joint_velocities = [state[1] for i, state in enumerate(joint_states) if i < 7 or i >= 9]  # Exclude joints 7 and 8

    # Treat joints 9-14 as a single joint (claw)
    claw_position = joint_states[9][0]  # Use joint 9 as the representative for the claw
    claw_velocity = joint_states[9][1]  # Use joint 9 as the representative for the claw

    # Replace positions and velocities for joints 9-14 with the claw position and velocity
    joint_positions = joint_positions[:7] + [claw_position]  # 7 motion joints + 1 claw joint
    joint_velocities = joint_velocities[:7] + [claw_velocity]  # 7 motion joints + 1 claw joint

    # Get end-effector position and orientation
    end_effector_pos, end_effector_orn = p.getLinkState(robot_id, 14)[:2]  # Assuming joint 14 is the end effector

    # Get object position and orientation
    object_pos, object_orn = p.getBasePositionAndOrientation(object1_id)

    # Combine into a single observation vector
    observation = np.concatenate([
        joint_positions,
        joint_velocities,
        end_effector_pos,
        end_effector_orn,
        object_pos,
        object_orn
    ])
    return observation
def apply_action(robot_id, action):
    # Action is a list of target positions for joints 0-6 and the claw (joints 9-14)
    for i in range(7):  # Joints 0-6
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=action[i],
            force=500  # Maximum force to apply
        )
    # Apply the same target position to all claw joints (9-14)
    claw_target = action[7]  # The 8th value in the action array is for the claw
    for i in range(9, 15):  # Joints 9-14
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=claw_target,
            force=500
        )
def compute_reward(robot_id, object1_id, object2_id, object3_id, home_position):
    """Compute the reward for the current state."""
    # Get end-effector position
    end_effector_pos = p.getLinkState(robot_id, 14)[0]  # Assuming joint 14 is end effector
    
    # Get object position
    object_pos = p.getBasePositionAndOrientation(object1_id)[0]
    
    # Calculate distance between end-effector and object
    distance = np.sqrt(
        (end_effector_pos[0] - object_pos[0])**2 +
        (end_effector_pos[1] - object_pos[1])**2 +
        (end_effector_pos[2] - object_pos[2])**2
    )
    
    # Initialize reward
    reward = 0
    
    # Distance-based reward (negative distance to encourage getting closer)
    # Scale factor can be adjusted to change the importance of distance
    distance_reward = -distance * 10
    reward += distance_reward
    
    # Check for vertical alignment (reward approaching from above)
    horizontal_distance = np.sqrt(
        (end_effector_pos[0] - object_pos[0])**2 +
        (end_effector_pos[1] - object_pos[1])**2
    )
    vertical_distance = end_effector_pos[2] - object_pos[2]
    
    # Reward being directly above the object
    if horizontal_distance < 0.1 and vertical_distance > 0:
        vertical_alignment_reward = 5.0 * (1.0 - horizontal_distance/0.1)
        reward += vertical_alignment_reward
    
    # Check for collisions with all objects
    # We'll check contacts for the robot's links with all objects
    collision_reward = 0
    for i in range(p.getNumJoints(robot_id)):
        # Check collision with object1
        contact_points = p.getContactPoints(robot_id, object1_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
            
        # Check collision with object2
        contact_points = p.getContactPoints(robot_id, object2_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
            
        # Check collision with object3
        contact_points = p.getContactPoints(robot_id, object3_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
    
    reward += collision_reward
    
    # Bonus reward for being very close to the object
    if distance < 0.05:  # 5cm threshold
        reward += 20.0  # Big bonus for being very close
    
    return reward



def is_object_grasped(robot_id, object1_id):
    """Check if the object is currently grasped by the robot."""
    # Check contact points between gripper fingers and object
    # Assuming joints 9-14 are the gripper fingers
    finger_joints = range(9, 15)
    
    contact_count = 0
    for joint in finger_joints:
        contacts = p.getContactPoints(robot_id, object1_id, joint)
        if contacts:
            contact_count += 1
    
    # Consider object grasped if multiple finger joints are in contact
    return contact_count >= 2

def is_at_home(current_pos, home_position, tolerance=0.05):
    """Check if the end effector is at the home position within tolerance."""
    distance = np.sqrt(
        (current_pos[0] - home_position[0])**2 +
        (current_pos[1] - home_position[1])**2 +
        (current_pos[2] - home_position[2])**2
    )
    return distance < tolerance
def get_observation(robot_id, object1_id):
    # Get robot joint states (joints 0-6 for motion, 9-14 for claw)
    joint_states = p.getJointStates(robot_id, range(15))  # All 15 joints
    joint_positions = [state[0] for i, state in enumerate(joint_states) if i < 7 or i >= 9]  # Exclude joints 7 and 8
    joint_velocities = [state[1] for i, state in enumerate(joint_states) if i < 7 or i >= 9]  # Exclude joints 7 and 8

    # Treat joints 9-14 as a single joint (claw)
    claw_position = joint_states[9][0]  # Use joint 9 as the representative for the claw
    claw_velocity = joint_states[9][1]  # Use joint 9 as the representative for the claw

    # Replace positions and velocities for joints 9-14 with the claw position and velocity
    joint_positions = joint_positions[:7] + [claw_position]  # 7 motion joints + 1 claw joint
    joint_velocities = joint_velocities[:7] + [claw_velocity]  # 7 motion joints + 1 claw joint

    # Get end-effector position and orientation
    end_effector_pos, end_effector_orn = p.getLinkState(robot_id, 14)[:2]  # Assuming joint 14 is the end effector

    # Get object position and orientation
    object_pos, object_orn = p.getBasePositionAndOrientation(object1_id)

    # Combine into a single observation vector
    observation = np.concatenate([
        joint_positions,
        joint_velocities,
        end_effector_pos,
        end_effector_orn,
        object_pos,
        object_orn
    ])
    return observation
def apply_action(robot_id, action):
    # Action is a list of target positions for joints 0-6 and the claw (joints 9-14)
    for i in range(7):  # Joints 0-6
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=action[i],
            force=500  # Maximum force to apply
        )
    # Apply the same target position to all claw joints (9-14)
    claw_target = action[7]  # The 8th value in the action array is for the claw
    for i in range(9, 15):  # Joints 9-14
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=claw_target,
            force=500
        )
def compute_reward(robot_id, object1_id, home_position):
    # Get end-effector position
    end_effector_pos = p.getLinkState(robot_id, 14)[0]  # Assuming joint 14 is end effector
    
    # Get object position
    object_pos = p.getBasePositionAndOrientation(object1_id)[0]
    
    # Calculate distance between end-effector and object
    distance = np.sqrt(
        (end_effector_pos[0] - object_pos[0])**2 +
        (end_effector_pos[1] - object_pos[1])**2 +
        (end_effector_pos[2] - object_pos[2])**2
    )
    
    # Initialize reward
    reward = 0
    
    # Distance-based reward (negative distance to encourage getting closer)
    # Scale factor can be adjusted to change the importance of distance
    distance_reward = -distance * 10
    reward += distance_reward
    
    # Check for vertical alignment (reward approaching from above)
    horizontal_distance = np.sqrt(
        (end_effector_pos[0] - object_pos[0])**2 +
        (end_effector_pos[1] - object_pos[1])**2
    )
    vertical_distance = end_effector_pos[2] - object_pos[2]
    
    # Reward being directly above the object
    if horizontal_distance < 0.1 and vertical_distance > 0:
        vertical_alignment_reward = 5.0 * (1.0 - horizontal_distance/0.1)
        reward += vertical_alignment_reward
    
    # Check for collisions with all objects
    # We'll check contacts for the robot's links with all objects
    collision_reward = 0
    for i in range(p.getNumJoints(robot_id)):
        # Check collision with object1
        contact_points = p.getContactPoints(robot_id, object1_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
            
        # Check collision with object2
        contact_points = p.getContactPoints(robot_id, env.object2_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
            
        # Check collision with object3
        contact_points = p.getContactPoints(robot_id, env.object3_id, i)
        if len(contact_points) > 0:
            collision_reward = -10
            break
    
    reward += collision_reward
    
    # Bonus reward for being very close to the object
    if distance < 0.05:  # 5cm threshold
        reward += 20.0  # Big bonus for being very close
    
    return reward
def is_done(robot_id, object1_id, home_position, steps):
    # Get object position and orientation
    object_pos, object_orn = p.getBasePositionAndOrientation(object1_id)
    
    # Get end-effector position
    end_effector_pos = p.getLinkState(robot_id, 14)[0]  # Assuming joint 14 is end effector
    
    # Check if object is grasped
    object_grasped = is_object_grasped(robot_id, object1_id)
    
    # Check if at home position (with some tolerance)
    at_home = is_at_home(end_effector_pos, home_position)
    
    # Get object height (to check if it's lifted)
    object_lifted = object_pos[2] > 0.1  # Assuming initial height is 0
    
    # Check maximum steps (optional timeout)
    max_steps = 1000  # Adjust this value based on your needs
    timeout = steps >= max_steps
    
    # Episode is done if:
    # 1. Object is grasped AND robot is at home position
    # 2. Maximum steps reached (timeout)
    done = (object_grasped and object_lifted and at_home) or timeout
    
    return done

def is_object_grasped(robot_id, object1_id):
    """Check if the object is currently grasped by the robot."""
    # Check contact points between gripper fingers and object
    # Assuming joints 9-14 are the gripper fingers
    finger_joints = range(9, 15)
    
    contact_count = 0
    for joint in finger_joints:
        contacts = p.getContactPoints(robot_id, object1_id, joint)
        if contacts:
            contact_count += 1
    
    # Consider object grasped if multiple finger joints are in contact
    # You might need to adjust this threshold based on your gripper
    return contact_count >= 2

def is_at_home(current_pos, home_position, tolerance=0.05):
    """Check if the end effector is at the home position within tolerance."""
    distance = np.sqrt(
        (current_pos[0] - home_position[0])**2 +
        (current_pos[1] - home_position[1])**2 +
        (current_pos[2] - home_position[2])**2
    )
    return distance < tolerance
# Connect to PyBullet in GUI mode
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane
planeId = p.loadURDF("plane.urdf")

# Load robot and object
robot_urdf_path = "/output.urdf"
robot_start_pos = [0, 0, 0]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
robot_id = p.loadURDF(robot_urdf_path, basePosition=robot_start_pos, baseOrientation=robot_start_orientation, useFixedBase=True)

object1_id = p.loadURDF("/housing.urdf", basePosition=[0.75, 0, 0])
object2_id = p.loadURDF("/housing.urdf", basePosition=[0.4, 0, 0])
object3_id = p.loadURDF("/housing.urdf", basePosition=[0.5, 0.2, 0])

# # Main loop for visualization
# while True:
#     # Step the simulation
#     p.stepSimulation()

#     # Add a small delay to visualize in real time
#     time.sleep(1. / 240.)



class RobotEnv:
    def __init__(self):
        self.robot_id = None
        self.object1_id = None
        self.home_position = [0, 0, 0]  # Define home position
        self.steps = 0

    def reset(self):
        # Reset the simulation
        p.resetSimulation()
        self.robot_id = p.loadURDF("/output.urdf", [0, 0, 0], useFixedBase=True)
        self.object1_id = p.loadURDF("housing.urdf", [0.75, 0, 0])
        self.object2_id = p.loadURDF("housing.urdf", [0.4, 0, 0])
        self.object3_id = p.loadURDF("housing.urdf", [0.5, 0.2, 0])
        self.steps = 0
        return get_observation(self.robot_id, self.object1_id)

    def step(self, action):
        apply_action(self.robot_id, action)
        p.stepSimulation()
        observation = get_observation(self.robot_id, self.object1_id)
        reward = compute_reward(self.robot_id, self.object1_id, self.home_position)
        done = is_done(self.robot_id, self.object1_id, self.home_position, self.steps)
        self.steps += 1
        return observation, reward, done, {}

# Create the environment
env = DummyVecEnv([lambda: RobotEnv()])

# Train the RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model
model.save("robot_arm_pickup")