import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchDiscrete import *
from agentTorchContinuous import *
import cProfile

#Params
name = "CartPoleDiscrete-5"
action_type = 'continuous'

max_timestep = 50000
learn_interval = 10*200
batch_size = 500
learn_iterations = 200
memory_buffer_size = 50000
discount = 0.999
value_learn_rate = 0.001
policy_learn_rate = value_learn_rate/20
policy_copy_rate = 1.0
next_learn_factor = 0.8
action_grad_max = 10000
save_interval = learn_interval

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

agent = AgentTorchDiscrete(name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
                           discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate,
                           policy_copy_rate=policy_copy_rate, next_learn_factor=next_learn_factor,
                           action_grad_max=action_grad_max,
                           debug=debug)

cumulative_score = 0
done = True

# pr = cProfile.Profile()
# pr.enable()

for timestep in range(1, max_timestep + 1):

    if done:
        score = 0
        episode_timestep = 0

        next_observation_partial = env.reset()
        next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
        if render: env.render()

    episode_timestep += 1
    observation = next_observation

    action = agent.act(observation)
    next_observation_partial, reward, done, info = env.step(action)
    next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
    agent.record(observation, action, reward, next_observation, done)
    score += reward

    if render: env.render()
    if render: time.sleep(delay)

    if done: print(
        "Episode " + str(timestep) + " finished after " + str(episode_timestep) + " timesteps - reward = " + str(score))

    if (timestep % learn_interval) == 0: agent.learn()
    if (timestep % save_interval) == 0: agent.save()

agent.save()

# pr.disable()
env.close()
# pr.dump_stats('profile.dat')