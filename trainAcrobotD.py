import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchDiscrete import AgentTorchDiscrete
import time

#Params
learn_episodes = 10000
test_episodes = 200
learn_interval = 10
render = False
delay = 0.0

# Run simulation
env = gym.make('Acrobot-v1')

num_of_action_values = [3]
state_space_min = [-1, -1, -1, -1, -12, -28] + [0]
state_space_max = [1, 1, 1, 1, 12, 28] + [500]
reward_space_min = [-1]
reward_space_max = [1]
agent = AgentTorchDiscrete("Acrobat1D", num_of_action_values, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           discount=0.999, batch_size=1000, epoch_size=10000, value_learn_rate=0.0001, policy_learn_rate=0.0001, policy_copy_rate=0.01,
                           learn_iterations=10, memory_buffer_size=10000000, next_learn_factor=0.5, debug=False)

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

        action = agent.act(observation, use_max_policy=testing)
        action = np.squeeze(action)
        next_observation_partial, reward, done, info = env.step(action)
        next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
        if n < learn_episodes:
            agent.record(observation, action, reward, next_observation, done, timestep)
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
