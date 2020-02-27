import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchDiscrete import AgentTorchDiscrete
import time
import quantize

#Params
learn_episodes = 10000
test_episodes = 100
learn_interval = 10
render = True
delay = 0.01

# Run simulation
env = gym.make('Walker2DBulletEnv-v0', render=render)

num_of_action_values = [3, 3, 3, 3, 3, 3]
state_space_min = [-1]*22 + [0]
state_space_max = [1]*22 + [1000]
reward_space_min = [0]
reward_space_max = [1]
agent = AgentTorchDiscrete("Walker11D", num_of_action_values, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           discount=0.999, batch_size=1000, value_learn_rate=0.001, policy_learn_rate=0.0001, policy_copy_rate=0.001,
                           learn_iterations=10, memory_buffer_size=10000, next_learn_factor=0.5)

cumulative_score = 0
for n in range(learn_episodes + test_episodes):

    learning = n < learn_episodes
    testing = not learning

    score = 0
    timestep = 0

    next_observation_partial = env.reset()
    next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
    if render: env.render()

    while True:

        timestep += 1
        observation = next_observation

        action_idx = agent.act(observation, use_max_policy=testing)
        action = quantize.unquantize(action_idx, -1, 1, 3)
        next_observation_partial, reward, done, info = env.step(action)
        next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
        if n < learn_episodes:
            agent.record(observation, action_idx, reward, next_observation, done, timestep)
        score += reward

        if render: env.render()

        if done:

            if learning:
                print("Learn episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                if (n % learn_interval) == (learn_interval-1) or (n+1) == learn_episodes: agent.learn()

            else:
                print("Test episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                cumulative_score += score

            break

        time.sleep(delay)

if test_episodes > 0:
    print("Test average score " + str(cumulative_score / test_episodes))

env.close()
