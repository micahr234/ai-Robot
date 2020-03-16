import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentContinuous import *
from run import *

#Params
name = "HopperContinuous-1"
action_type = 'continuous'

max_timestep = 1000000
learn_interval = 10*1000
batch_size = 500
learn_iterations = int(100000/batch_size)
memory_buffer_size = 500000
discount = 0.999
value_learn_rate = 0.0005
policy_learn_rate = value_learn_rate/80
policy_copy_rate = 1.0
next_learn_factor = 0.1
save_interval = learn_interval

num_of_action_values = [9, 9, 9] # For discrete environments only
action_space_min = [-1]*3
action_space_max = [1]*3
state_space_min = [-1]*15 + [1]
state_space_max = [1]*15 + [1000]
reward_space_min = [-1]
reward_space_max = [1]

render = True
delay = 0.0
debug = False

# Run simulation
env = gym.make('HopperBulletEnv-v0', render=render)

agent = AgentContinuous(name, action_type, None, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                        batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                        discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, policy_copy_rate=policy_copy_rate, next_learn_factor=next_learn_factor,
                        debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=debug)