import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentDiscrete import *
from run import *
from utils import *

#Params
name = "HopperDiscrete-1"
action_type = 'continuous'

max_timestep = 10000000
learn_interval = 2000
batch_size = 500
learn_iterations = int(100000*4/batch_size)
memory_buffer_size = 500000
discount = 1.0
value_learn_rate = 0.0003
policy_learn_rate = 0.0001
next_learn_factor = 0.8
save_interval = learn_interval*10
noise_power = 0.0

num_of_action_values = [9, 9, 9] # For continuous environments
action_space_min = [-1]*3
action_space_max = [1]*3
state_space_min = [-1]*15 + [1]
state_space_max = [1]*15 + [1000]
reward_space_min = [-1]
reward_space_max = [1]

render = False
delay = 0.0
debug = False
profile = False

# Run simulation
env = gym.make('HopperBulletEnv-v0', render=render)

agent = AgentDiscrete(name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                      batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                      discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, next_learn_factor=next_learn_factor,
                      debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=profile, enable_eposide_timestep=True, noise_power=noise_power)