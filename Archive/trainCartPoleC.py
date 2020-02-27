import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchContinuous import AgentTorchContinuous

#Params
learn_episodes = 1
test_episodes = 100
learn_interval = 100
render = False

# Run simulation
env = gym.make('CartPoleBulletEnv-v1', renders=render, discrete_actions=False)

action_space_min = [-10]
action_space_max = [10]
state_space_min = [-1, -1, -1, -1, 0]
state_space_max = [1, 1, 1, 1, 200]
reward_space_min = [0]
reward_space_max = [1]
agent = AgentTorchContinuous("Cart2C", action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           discount=0.999, batch_size=1000, value_learn_rate=0.001, policy_learn_rate=0.0001, policy_copy_rate=0.1,
                           learn_iterations=1000, memory_buffer_size=1000000, next_learn_factor=0.1)

cumulative_score = 0
for n in range(learn_episodes + test_episodes):

    score = 0
    timestep = 0

    next_observation_partial = env.reset()
    next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
    if render: env.render()

    while True:

        timestep += 1
        observation = next_observation

        action = agent.act(observation)
        next_observation_partial, reward, done, info = env.step(action)
        next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
        if n < learn_episodes:
            agent.record(observation, action, reward, next_observation, done, timestep)
        score += reward

        if render: env.render()

        if done:

            if n < learn_episodes:
                print("Learn episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                if (n % learn_interval) == (learn_interval-1) or (n+1) == learn_episodes: agent.learn()

            else:
                print("Test episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                cumulative_score += score

            break

if test_episodes > 0:
    print("Test average score " + str(cumulative_score / test_episodes))

env.close()
