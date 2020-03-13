import numpy as np
import gym
import pybullet
import pybullet_envs
import pybullet_data
from agentTorchDiscrete import *
from agentTorchContinuous import *
import cProfile

#Params
name = "CartPoleDiscrete-11"
action_type = 'continuous'

learn_episodes = 300
test_episodes = 0
learn_interval = 10
batch_size = 500
learn_iterations = learn_interval*20
memory_buffer_size = batch_size*100
discount = 0.999
value_learn_rate = 0.001
policy_learn_rate = value_learn_rate/20
policy_copy_rate = 1.0
next_learn_factor = 0.8
action_grad_max = 10000
save_frequency = learn_interval

#num_of_action_values = [2] # For discrete environments
num_of_action_values = [5000] # For continuous environments
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

#pr = cProfile.Profile()
#pr.enable()

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

#pr.disable()

if test_episodes > 0:
    print("Test average score " + str(cumulative_score / test_episodes))

env.close()

#pr.dump_stats('profile.dat')
