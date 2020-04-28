import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentContinuous import *
from run import *

#Params
name = "CartPoleContinuous-9"

max_timestep = 100000
learn_interval = 2000
batch_size = 500
learn_iterations = int(learn_interval*50/batch_size)
memory_buffer_size = max_timestep*10
discount = 1.0
value_learn_rate = 0.001/4
policy_learn_rate = 0.001/3
preprocess_learn_rate = 0.001
policy_delay = 10
next_learn_factor = 0.8
randomness = 0.1
save_interval = max_timestep+1
noise_power = 0.0
value_hidden_layer_sizes = [256, 128]
policy_hidden_layer_sizes = [256, 128]
preprocess_hidden_layer_sizes=[20, 10]
num_of_actions = 1
num_of_states = 4 + 1

render = False
delay = 0.0
debug = False
profile = False

# Run simulation
env = gym.make('CartPoleBulletEnv-v1', renders=render, discrete_actions=False)

agent = AgentContinuous(name,  num_of_actions, num_of_states,
                        value_hidden_layer_sizes = value_hidden_layer_sizes, policy_hidden_layer_sizes = policy_hidden_layer_sizes, preprocess_hidden_layer_sizes=preprocess_hidden_layer_sizes,
                        batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                        discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, preprocess_learn_rate=preprocess_learn_rate, policy_delay=policy_delay,
                        next_learn_factor=next_learn_factor, randomness=randomness,
                        debug=debug)

Run(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, profile=profile, enable_eposide_timestep=True, noise_power=noise_power)