import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchContinuous import *

#Params
learn_episodes = 10000
test_episodes = 200
learn_interval = 10
render = True
delay = 0.0
save_frequency = 10

# Run simulation
env = gym.make('HopperBulletEnv-v0', render=render)

action_space_min = [-1]*3
action_space_max = [1]*3
state_space_min = [-1]*15 + [1]
state_space_max = [1]*15 + [1000]
reward_space_min = [-1]
reward_space_max = [1]
agent = AgentTorchContinuous("HopperC1", action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                           batch_size=1000, learn_iterations=10, memory_buffer_size=1000*20,
                           discount=0.99, value_learn_rate=0.0001, policy_learn_rate=0.00001, policy_copy_rate=0.001, next_learn_factor=0.3,
                           debug=False)

cumulative_score = 0
for n in range(1, learn_episodes + test_episodes + 1):

    learning = n <= learn_episodes
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
        next_observation_partial, reward, done, info = env.step(action)
        next_observation = np.concatenate((next_observation_partial, np.array(timestep, ndmin=1)))
        if learning:
            agent.record(observation, action, reward, next_observation, done)
        score += reward

        if render: env.render()

        if done:

            if learning:
                print("Learn episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                if ((n % learn_interval) == 0) or n == learn_episodes: agent.learn()
                if ((n % save_frequency) == 0) or n == learn_episodes: agent.save()

            else:
                print("Test episode " + str(n) + " finished after " + str(timestep) + " timesteps - reward = " + str(score))
                cumulative_score += score

            break

        time.sleep(delay)

if test_episodes > 0:
    print("Test average score " + str(cumulative_score / test_episodes))

env.close()
