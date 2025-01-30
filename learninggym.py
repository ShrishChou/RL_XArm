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

def is_done(robot_id, object1_id, home_position, steps):
    """Determine if the episode should end."""
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
    return contact_count >= 2

def is_at_home(current_pos, home_position, tolerance=0.05):
    """Check if the end effector is at the home position within tolerance."""
    distance = np.sqrt(
        (current_pos[0] - home_position[0])**2 +
        (current_pos[1] - home_position[1])**2 +
        (current_pos[2] - home_position[2])**2
    )
    return distance < tolerance
class RobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print("Initializing RobotEnv...")
        self.robot_urdf = "output.urdf"
        self.housing_urdf = "housing.urdf"
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0], dtype=np.float32),
            high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 0.85], dtype=np.float32)
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32
        )
        
        # Initialize simulation
        try:
            print("Connecting to PyBullet...")
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            print("Connected to PyBullet successfully")
            
            # Print current working directory and available files
            print(f"Current working directory: {os.getcwd()}")
            print("Checking for URDF files...")
            
            # Check if URDF files exist
            urdf_files = {
                'robot': "output.urdf",
                'housing': "housing.urdf"
            }
            
            for name, path in urdf_files.items():
                if os.path.exists(path):
                    print(f"Found {name} URDF at: {path}")
                else:
                    print(f"WARNING: Cannot find {name} URDF at: {path}")
            
            # Load plane
            plane_id = p.loadURDF("plane.urdf")
            print("Loaded plane successfully")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
        
        self.robot_id = None
        self.object1_id = None
        self.object2_id = None
        self.object3_id = None
        self.home_position = [0, 0, 0]
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: The seed for random number generation
            options: Additional options for reset (not used currently)
            
        Returns:
            observation: The initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        print("Resetting environment...")
        
        try:
            p.resetSimulation()
            
            # Load robot
            print(f"Loading robot URDF from: {self.robot_urdf}")
            self.robot_id = p.loadURDF(self.robot_urdf, [0, 0, 0], useFixedBase=True)
            if self.robot_id is None:
                raise Exception("Failed to load robot URDF")
            print(f"Robot loaded successfully with ID: {self.robot_id}")
            
            # Load objects
            print("Loading object URDFs...")
            self.object1_id = p.loadURDF(self.housing_urdf, [0.75, 0, 0])
            if self.object1_id is None:
                raise Exception("Failed to load object1 URDF")
            print(f"Object 1 loaded successfully with ID: {self.object1_id}")
            
            self.object2_id = p.loadURDF(self.housing_urdf, [0.4, 0, 0])
            self.object3_id = p.loadURDF(self.housing_urdf, [0.5, 0.2, 0])
            
            # Get and verify observation
            print("Getting initial observation...")
            observation = get_observation(self.robot_id, self.object1_id)
            observation = np.array(observation, dtype=np.float32)
            
            if observation.shape != self.observation_space.shape:
                raise ValueError(f"Observation shape mismatch. Expected {self.observation_space.shape}, got {observation.shape}")
            
            self.steps = 0
            print("Reset completed successfully")
            return observation, {}
            
        except Exception as e:
            print(f"Error during reset: {str(e)}")
            raise
    def step(self, action):
        try:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            apply_action(self.robot_id, action)
            p.stepSimulation()
            
            observation = get_observation(self.robot_id, self.object1_id)
            observation = np.array(observation, dtype=np.float32)
            
            # Pass all object IDs to compute_reward
            reward = compute_reward(
                self.robot_id, 
                self.object1_id, 
                self.object2_id, 
                self.object3_id, 
                self.home_position
            )
            
            done = is_done(self.robot_id, self.object1_id, self.home_position, self.steps)
            
            self.steps += 1
            return observation, reward, done, False, {}
            
        except Exception as e:
            print(f"Error during step: {str(e)}")
            raise

    def render(self):
        pass  # PyBullet already renders in GUI mode

    def close(self):
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            print("Disconnected from PyBullet")
# First unregister if it exists (in case of re-running)
try:
    if 'RobotEnv-v0' in gym.envs.registry:
        del gym.envs.registry['RobotEnv-v0']
except Exception as e:
    print(f"Warning during environment cleanup: {e}")

# Register the environment
print("Registering environment...")
register(
    id='RobotEnv-v0',
    entry_point='__main__:RobotEnv',
    max_episode_steps=1000,
)
print("Environment registered successfully")

try:
    # Create the environment
    print("Creating base environment...")
    env = gym.make('RobotEnv-v0', disable_env_checker=True)
    print("Base environment created")
    
    # Test reset before vectorization
    print("Testing initial reset...")
    initial_obs, _ = env.reset()
    print(f"Initial observation shape: {initial_obs.shape}")
    
    # Wrap in DummyVecEnv
    print("Creating vector environment...")
    vec_env = DummyVecEnv([lambda: env])
    print("Vector environment created")
    
    # Create the model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
    )
    print("PPO model created successfully")
    
    # Start training
    print("Starting training...")
    model.learn(
        total_timesteps=100000,
        progress_bar=True,
        log_interval=10
    )
    print("Training completed")
    
    # Save the model
    print("Saving model...")
    model.save("robot_arm_pickup")
    print("Model saved successfully")
    
    # Clean up
    env.close()
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Print the full error traceback
    import traceback
    traceback.print_exc()
    
    # Clean up in case of error
    try:
        env.close()
    except:
        pass
    
register(
    id='RobotEnv-v0',
    entry_point='__main__:RobotEnv',
)


# Create and wrap the environment
env = gym.make('RobotEnv-v0', disable_env_checker=True)
env = DummyVecEnv([lambda: env])