import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentDiscrete import *
from run import *
from utils import *

#Params
name = "CartPoleDiscrete-125"
action_type = 'continuous'

max_timestep = 100000
learn_interval = 2000
batch_size = 500
learn_iterations = int(learn_interval*50/batch_size)
memory_buffer_size = max_timestep*10
discount = 1.0
value_learn_rate = 0.001/4
policy_learn_rate = 0.001/3
policy_delay = 10
next_learn_factor = 0.8
randomness = 0.3
save_interval = max_timestep+1
noise_power = 0.0
value_hidden_layer_sizes = [512, 256, 128, 64]
policy_hidden_layer_sizes = [256, 128, 64]

#num_of_action_values = [2] # For discrete environments
num_of_action_values = [100] # For continuous environments
action_space_min = [-10]
action_space_max = [10]
state_space_min = [-1, -1, -1, -1] + [1]
state_space_max = [1, 1, 1, 1] + [200]
reward_space_min = [0]
reward_space_max = [1]

render = False
delay = 0.0
debug = False
profile = False

# Run simulation
discrete_actions = True if action_type == 'discrete' else False
env = gym.make('CartPoleBulletEnv-v1', renders=render, discrete_actions=discrete_actions)

agent = AgentDiscrete(name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                      value_hidden_layer_sizes = value_hidden_layer_sizes, policy_hidden_layer_sizes = policy_hidden_layer_sizes,
                      batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                      discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, policy_delay=policy_delay, next_learn_factor=next_learn_factor, randomness=randomness,
                      debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=profile, enable_eposide_timestep=True, noise_power=noise_power)