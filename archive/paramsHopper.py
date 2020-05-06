import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentContinuous import *
from run import *

#Params
name = "HopperContinuous-9"

max_timestep = 10000000
learn_interval = 2000
batch_size = 500
learn_iterations = int(learn_interval*50/batch_size)
memory_buffer_size = 500000
discount = 1.0
value_learn_rate = 0.001/3 * 0.25
policy_learn_rate = 0.001/3 * 0.25
policy_delay = 10
next_learn_factor = 0.8
randomness = 0.1
save_interval = learn_interval
noise_power = 0.0
value_hidden_layer_sizes = [512, 256, 128]
policy_hidden_layer_sizes = [256, 128, 64]

render = False
delay = 0.0
debug = False
profile = False

# Run simulation
env = gym.make('HopperBulletEnv-v0', render=render)

agent = AgentContinuous(name, value_hidden_layer_sizes = value_hidden_layer_sizes, policy_hidden_layer_sizes = policy_hidden_layer_sizes,
                        batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                        discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, policy_delay=policy_delay, next_learn_factor=next_learn_factor, randomness=randomness,
                        debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=profile, enable_eposide_timestep=True, noise_power=noise_power)