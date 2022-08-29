import numpy as np
from robotman import Manipulator2D

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

# Call the robotic arm environment
env = Manipulator2D()
# Get how many actions do we need from the environment variable
n_actions = env.action_space.shape[-1]

# Select DDPG algorithm from the stable-baseline library
model = PPO('MlpPolicy', env=env, verbose=1)

# Train an RL agent for 400,000 timesteps
# "learn" method calls "step" method in an environment
print('checking')
#model.learn(total_timesteps=100000, callback=eval_callback)
model.learn(total_timesteps=1000000)

# Save the weights
# Additional assignment : How do I save the policy network that returned the best reward value during training, not the result of learning over 400,000 timesteps?
# Tip : Let's use the callback function for the learn method
model.save("ppo_manipulator2D")

# Delete the model variable from the memory
del model  # remove to demonstrate saving and loading

# Load the weights from the saved training file
model = PPO.load("ppo_manipulator2D")

# Reset the simulation environment
obs = env.reset()

while True:
    # The trained model returns action values using the current observation
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    cnt=0

    if done:
        break

env.render()