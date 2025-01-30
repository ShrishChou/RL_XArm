import math
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO
from gymnasium import spaces
from gymnasium.envs.registration import register
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HumanOutputFormat

def get_observation(robot_id, object1_id):
    """Get simplified observation focused on positioning task."""
    # Get positions for joints 0-6 only
    joint_states = [p.getJointState(robot_id, i) for i in range(7)]
    joint_positions = [state[0] for state in joint_states]
    
    # Get end-effector position and orientation
    end_effector_state = p.getLinkState(robot_id, 14)
    end_effector_pos = end_effector_state[0]
    end_effector_orn = p.getEulerFromQuaternion(end_effector_state[1])
    
    # Get object position
    object_pos, _ = p.getBasePositionAndOrientation(object1_id)
    
    # Calculate relative position (end effector - object)
    relative_pos = np.array(end_effector_pos) - np.array(object_pos)
    
    # Combine into observation vector:
    # 7 joint positions + 3 relative positions + 3 end effector orn + 3 object pos = 16 total
    observation = np.concatenate([
        joint_positions,     # 7
        relative_pos,       # 3 (relative position)
        end_effector_orn,   # 3 (roll, pitch, yaw)
        object_pos,         # 3
    ]).astype(np.float32)
    
    return observation

def apply_action(robot_id, action):
    """Apply actions as end-effector position changes."""
    # Scale actions from [-1, 1] to actual position changes
    position_change = np.array(action) * 0.05  # Scale factor of 0.05 meters
    
    # Get current end-effector state
    end_effector_state = p.getLinkState(robot_id, 14)
    current_pos = end_effector_state[0]
    
    # Compute target position
    target_pos = [
        current_pos[0] + position_change[0],
        current_pos[1] + position_change[1],
        current_pos[2] + position_change[2]
    ]
    
    # Always maintain vertical orientation
    target_orn = p.getQuaternionFromEuler([0, 0, 0])
    
    # Use inverse kinematics
    joint_angles = p.calculateInverseKinematics(
        robot_id,
        endEffectorLinkIndex=14,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    # Apply joint angles with position control
    for i in range(7):
        p.setJointMotorControl2(
            robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_angles[i],
            force=500,
            maxVelocity=1.0
        )

def compute_reward(robot_id, object1_id):
    """Compute reward based on position above object and horizontal orientation of joint 6."""
    # Get positions and orientation
    link_state = p.getLinkState(robot_id, 6,computeForwardKinematics=True)
    position, orientation = link_state[4], link_state[5]
    
    # Get the rotation matrix of the end effector
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    
    forward_vector = rotation_matrix[:, 2]  # First column of the rotation matrix

    # Define the vertical axis (Z-axis in PyBullet)
    vertical_axis = np.array([0, 0, 1])
    if forward_vector[2]<0:
        # Calculate the dot product between the forward vector and the vertical axis
        dot_product = np.dot(forward_vector, vertical_axis)

        # Calculate the angle between the forward vector and the vertical axis
        angle_from_vertical = np.arccos(dot_product)  # Result in radians
        angle_from_vertical_deg = abs(180-np.degrees(angle_from_vertical))  # Convert to degrees
    else:
        angle_from_vertical_deg=180
    print(f"Angle from vertical: {angle_from_vertical_deg} degrees")
    object1_position, _ = p.getBasePositionAndOrientation(object1_id)
    object1_position = np.array(object1_position)
    horizontal_dist = np.linalg.norm(position[:2] - object1_position[:2])
    # Smoother position reward with exponential decay
    position_reward = np.exp(-2.0 * horizontal_dist)
    
    # Smoother orientation reward with exponential decay
    if angle_from_vertical_deg < 5.0:  # Threshold for being "vertical"
        orientation_reward = 10.0  # High reward for being vertical
    else:
        orientation_reward = -angle_from_vertical_deg
    
    # Additional rewards for improvement
    reward = position_reward + orientation_reward
    
    # Bonus for being close and oriented correctly
    if horizontal_dist < 0.2:
        reward += 2.0
        if horizontal_dist < 0.1:
            reward += 3.0
            if angle_from_vertical < 0.2:
                reward += 5.0
                if angle_from_vertical < 0.1:
                    reward += 10.0
    
    return float(reward)
    
    # return float(total_reward)

def is_done(robot_id, object1_id, steps):
    """Check if episode should end."""
    max_steps = 500
    
    link_state = p.getLinkState(robot_id, 6,computeForwardKinematics=True)
    position, orientation = link_state[4], link_state[5]
    
    # Get the rotation matrix of the end effector
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    
    forward_vector = rotation_matrix[:, 2]  # First column of the rotation matrix

    # Define the vertical axis (Z-axis in PyBullet)
    vertical_axis = np.array([0, 0, 1])
    if forward_vector[2]<0:
        # Calculate the dot product between the forward vector and the vertical axis
        dot_product = np.dot(forward_vector, vertical_axis)

        # Calculate the angle between the forward vector and the vertical axis
        angle_from_vertical = np.arccos(dot_product)  # Result in radians
        angle_from_vertical_deg = abs(180-np.degrees(angle_from_vertical))  # Convert to degrees
    else:
        angle_from_vertical_deg=180
    print(f"Angle from vertical: {angle_from_vertical_deg} degrees")
    object1_position, _ = p.getBasePositionAndOrientation(object1_id)
    object1_position = np.array(object1_position)
    horizontal_dist = np.linalg.norm(position[:2] - object1_position[:2])
    # Smoother position reward with exponential decay
    position_reward = np.exp(-2.0 * horizontal_dist)
    
    # Smoother orientation reward with exponential decay
    orientation_reward = np.exp(-2.0 * angle_from_vertical_deg)
    
    # Additional rewards for improvement
    reward = position_reward + orientation_reward
    # Success conditions:
    # 1. End effector is above object (within 10cm horizontally)
    # 2. Orientation is nearly vertical (within ~5.7 degrees)
    success = (horizontal_dist < 0.1 and angle_from_vertical_deg < 0.1)
    orientation_failure = angle_from_vertical_deg > 30  # ~28.6 degrees from vertical
    position_failure = position[2] < 0.1  # Too close to ground
    max_steps_reached = steps >= max_steps
    
    # Print reason for episode end
    if success or orientation_failure or position_failure or max_steps_reached:
        if success:
            print("\nEpisode ended: Success! Robot reached target position and orientation")
        elif orientation_failure:
            print(f"\nEpisode ended: Orientation failure (error: {angle_from_vertical_deg:.3f})")
        elif position_failure:
            print(f"\nEpisode ended: Too close to ground (height: {position[2]:.3f})")
        elif max_steps_reached:
            print("\nEpisode ended: Maximum steps reached")
    
    return success or max_steps_reached or orientation_failure or position_failure

class RobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action space: 3D position changes for end-effector
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),  # x, y, z position changes
            dtype=np.float32
        )
        
        # Observation space: 16 dimensions
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 16, dtype=np.float32),
            high=np.array([np.inf] * 16, dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize simulation
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        self.robot_urdf = "output.urdf"
        self.housing_urdf = "housing.urdf"
        self.steps = 0
        self.robot_id = None
        self.object1_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        
        # Set physics parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        
        # Load ground plane
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, 
                        lateralFriction=1.0,
                        restitution=0.1)
        
        # Load robot in a fixed starting position
        self.robot_id = p.loadURDF(self.robot_urdf, [0, 0, 0], useFixedBase=True)
        
        # Initialize joints
        for i in range(15):
            if i < 7:  # Main arm joints
                # Start in a more upright position
                start_pos = 0.0 if i == 0 else np.random.uniform(-0.2, 0.2)
                p.resetJointState(self.robot_id, i, targetValue=start_pos)
            else:  # Other joints (gripper, etc.)
                p.resetJointState(self.robot_id, i, targetValue=0.0)
        
        # Load object with random position
        if seed is not None:
            np.random.seed(seed)
        x = np.random.uniform(0.2, 0.7)  # Reduced range for easier learning
        y = np.random.uniform(-0.2, 0.2)
        self.object1_id = p.loadURDF(self.housing_urdf, [x, y, 0])
        
        # Set object dynamics
        p.changeDynamics(self.object1_id, -1,
                        mass=1.0,
                        lateralFriction=1.0,
                        restitution=0.1,
                        linearDamping=0.1,
                        angularDamping=0.1)
        
        # Let simulation settle
        for _ in range(100):
            p.stepSimulation()
        
        self.steps = 0
        observation = get_observation(self.robot_id, self.object1_id)
        return observation, {}

    def step(self, action):
        self.steps += 1
        
        # Apply action and simulate with slower steps
        apply_action(self.robot_id, action)
        for _ in range(30):  # Even more steps for slower simulation
            p.stepSimulation()
            time.sleep(0.02)  # Increased delay to 20ms between steps
        
        # Get state and compute reward
        observation = get_observation(self.robot_id, self.object1_id)
        reward = compute_reward(self.robot_id, self.object1_id)
        done = is_done(self.robot_id, self.object1_id, self.steps)
        
        return observation, reward, done, False, {}

    def close(self):
        p.disconnect(self.physics_client)

class TrainingInfoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        # Get the robot and object IDs from the environment
        # Access the underlying environment through the VecEnv and TimeLimit wrappers
        env = self.training_env.envs[0].env.env
        robot_id = env.robot_id
        
        link_state = p.getLinkState(robot_id, 6,computeForwardKinematics=True)
        position, orientation = link_state[4], link_state[5]
        
        # Get the rotation matrix of the end effector
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        
        forward_vector = rotation_matrix[:, 2]  # First column of the rotation matrix

        # Define the vertical axis (Z-axis in PyBullet)
        vertical_axis = np.array([0, 0, 1])
        if forward_vector[2]<0:
            # Calculate the dot product between the forward vector and the vertical axis
            dot_product = np.dot(forward_vector, vertical_axis)

            # Calculate the angle between the forward vector and the vertical axis
            angle_from_vertical = np.arccos(dot_product)  # Result in radians
            angle_from_vertical_deg = abs(180-np.degrees(angle_from_vertical))  # Convert to degrees
        else:
            angle_from_vertical_deg=180
        print(f"Angle from vertical: {angle_from_vertical_deg} degrees")
        
        
        
        # Get the reward for the current step
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Display current step info
        if self.n_calls % 10 == 0:  # Update every 10 steps
            print(f"\rStep: {self.n_calls}")
            print(f"Current Orientation (deg) - {angle_from_vertical_deg}")
            print(f"Target Orientation (deg)  - 0")
            print(f"Current Reward: {reward:.2f}")
            print("-" * 80)

        # If episode is done, log the episode info
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            avg_reward = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
            
            print(f"\nEpisode {self.episode_count} finished:")
            print(f"  Final Orientation (deg) - Roll: {angle_from_vertical_deg}")
            print(f"  Total Episode Reward: {self.current_episode_reward:.2f}")
            print(f"  Average Reward (last 100): {avg_reward:.2f}")
            print("=" * 80)
            
            # Reset episode reward
            self.current_episode_reward = 0

        return True

def main():
    try:
        if p.getConnectionInfo()['isConnected']:
            p.disconnect()
        
        if 'RobotEnv-v0' in gym.envs.registry:
            del gym.envs.registry['RobotEnv-v0']
        register(
            id='RobotEnv-v0',
            entry_point='__main__:RobotEnv',
            max_episode_steps=500,
        )

        env = gym.make('RobotEnv-v0', disable_env_checker=True)
        vec_env = DummyVecEnv([lambda: env])

        # Create PPO model with more verbose output
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=1e-3,  # Increased learning rate
            n_steps=1024,        # Shorter rollout for more frequent updates
            batch_size=128,      # Increased batch size
            n_epochs=5,          # Reduced epochs to prevent overfitting
            gamma=0.98,          # Slightly reduced discount factor
            clip_range=0.4,      # Increased clip range for more exploration
            ent_coef=0.01,       # Added entropy coefficient for exploration
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])],  # Simpler network
                activation_fn=torch.nn.ReLU
            ),
            tensorboard_log="./robot_tensorboard/"
        )

        # Create callback
        callback = TrainingInfoCallback()

        print("Starting training...")
        print("-------------------")
        
        model.learn(
            total_timesteps=100000,
            callback=callback,
            progress_bar=False
        )
        
        model.save("ppo_robot_positioning")

    except Exception as e:
        print(f"Training error: {e}")
    finally:
        if 'env' in locals():
            env.close()
        if p.getConnectionInfo()['isConnected']:
            p.disconnect()

if __name__ == "__main__":
    main()