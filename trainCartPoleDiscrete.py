import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentDiscrete import *
from run import *

#Params
name = "CartPoleDiscrete-6"
action_type = 'continuous'

max_timestep = 50000
learn_interval = 2000
batch_size = 500
learn_iterations = int(100000/batch_size)
memory_buffer_size = 10000
discount = 0.999
value_learn_rate = 0.001
policy_learn_rate = value_learn_rate/20
next_learn_factor = 0.8
save_interval = max_timestep+1
exploration_factor = 0.2

#num_of_action_values = [2] # For discrete environments
num_of_action_values = [3] # For continuous environments
action_space_min = [-10]
action_space_max = [10]
state_space_min = [-1, -1, -1, -1, 0]
state_space_max = [1, 1, 1, 1, 200]
reward_space_min = [0]
reward_space_max = [1]

render = False
delay = 0.0
debug = False

# Run simulation
discrete_actions = True if action_type == 'discrete' else False
env = gym.make('CartPoleBulletEnv-v1', renders=render, discrete_actions=discrete_actions)

agent = AgentDiscrete(name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                      batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size, exploration_factor=exploration_factor,
                      discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, next_learn_factor=next_learn_factor,
                      debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=debug)