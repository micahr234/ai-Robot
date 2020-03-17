import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentDiscrete import *
from run import *

#Params
name = "HopperDiscrete-10"
action_type = 'continuous'

max_timestep = 10000000
learn_interval = 2000
batch_size = 500
learn_iterations = int(100000/batch_size)
memory_buffer_size = 10000000
discount = 0.999
value_learn_rate = 0.001
policy_learn_rate = 0.0001
next_learn_factor = 0.8
save_interval = 50000
exploration_factor = 0.2
noise_power = 0.3

num_of_action_values = [3, 3, 3] # For continuous environments
action_space_min = [-1]*3
action_space_max = [1]*3
state_space_min = [-1]*15 # + [1]
state_space_max = [1]*15 # + [1000]
reward_space_min = [-1]
reward_space_max = [1]

render = True
delay = 0.0
debug = False

# Run simulation
env = gym.make('HopperBulletEnv-v0', render=render)

agent = AgentDiscrete(name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                      batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size, exploration_factor=exploration_factor,
                      discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, next_learn_factor=next_learn_factor,
                      debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=debug, enable_eposide_timestep=False)