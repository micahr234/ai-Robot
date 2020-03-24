import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentContinuous import *
from run import *

#Params
name = "CartPoleContinuous-11"
action_type = 'continuous'

max_timestep = 50000
learn_interval = 2000
batch_size = 1000 # increasing batch size give more exploration
learn_iterations = int(100000/batch_size)
memory_buffer_size = 50000
discount = 1.0
value_learn_rate = 0.001
policy_learn_rate = 0.00005
next_learn_factor = 0.3
save_interval = max_timestep+1
noise_power = 0.0

num_of_action_values = [1]
action_space_min = [-10]
action_space_max = [10]
state_space_min = [-1, -1, -1, -1] #+ [1]
state_space_max = [1, 1, 1, 1] #+ [1000]
reward_space_min = [0]
reward_space_max = [1]

render = False
delay = 0.0
debug = False
profile = False

# Run simulation
discrete_actions = True if action_type == 'discrete' else False
env = gym.make('CartPoleBulletEnv-v1', renders=render, discrete_actions=discrete_actions)

agent = AgentContinuous(name, action_type, None, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                        batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                        discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, next_learn_factor=next_learn_factor,
                        debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=profile, enable_eposide_timestep=False, noise_power=noise_power)