import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentContinuous import *
from run import *

#Params
name = "CartPoleContinuous-10"
action_type = 'continuous'

max_timestep = 50000
learn_interval = 10*200
batch_size = 1000
learn_iterations = int(100000/batch_size)
memory_buffer_size = 50000
discount = 0.999
value_learn_rate = 0.0001
value_copy_rate = 1.0
policy_learn_rate = value_learn_rate/100
policy_copy_rate = 1.0
next_learn_factor = 1.0
save_interval = learn_interval

num_of_action_values = [1]
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

agent = AgentContinuous(name, action_type, None, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                        batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                        discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate,
                        value_copy_rate=value_copy_rate, policy_copy_rate=policy_copy_rate, next_learn_factor=next_learn_factor,
                        debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=debug)