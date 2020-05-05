import argparse
import cProfile
import os
import time
import numpy as np

import gym
import pybullet
import pybullet_envs
import pybullet_data

from agentContinuous import AgentContinuous

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='Instance name for this run - leave blank for new random name',  type=str, default='', required=False)
parser.add_argument('-e', '--environment', help='Name of gym environment', type=str, required=True)
parser.add_argument('-a', '--agent', help='Select RL agent to use: "continuous"', type=str, required=True)
parser.add_argument('-p', '--param', help='Parameter filename', type=str, required=True)
parser.add_argument('-t', '--time', help='Time code and generate output in browser: "True" or  "False"', type=bool, default=True, required=False)
parser.add_argument('-r', '--render', help='Render environment', dest='render', action='store_true', required=False)
parser.add_argument('-rd', '--render_delay', help='Time to wait in between action updates', type=float, default=0.0, required=False)
parser.add_argument('-v', '--verbosity', help='Log verbosity in tensorboard', dest='verbosity', action='store_true', required=False)

args = parser.parse_args()

delay = args.render_delay
render = args.render
verbosity = args.verbosity

if args.name == '':
    args.name = args.agent + args.environment + str(time.time())
print('Run name: ' + args.name)
print('')

print('Building environment: ' + args.environment)
if args.environment == 'CartPoleBulletEnv-v1':
    env = gym.make(args.environment, renders=render, discrete_actions=False)
else:
    env = gym.make(args.environment)
print('')

print('Loading params from filename: ' + args.param)
exec(open(args.param).read())
print('')

print('Creating agent: ' + args.agent)
if args.agent == 'continuous':
    agent = AgentContinuous(args.name,  num_of_actions, num_of_states,
    value_hidden_layer_sizes = value_hidden_layer_sizes, policy_hidden_layer_sizes = policy_hidden_layer_sizes, preprocess_hidden_layer_sizes=preprocess_hidden_layer_sizes,
    batch_size=batch_size, learn_iterations=learn_iterations, memory_buffer_size=memory_buffer_size,
    discount=discount, value_learn_rate=value_learn_rate, policy_learn_rate=policy_learn_rate, preprocess_learn_rate=preprocess_learn_rate, policy_delay=policy_delay,
    next_learn_factor=next_learn_factor, randomness=randomness,
    verbosity=args.verbosity)
print('')


def Loop(env, agent, max_timestep, learn_interval, save_interval, render=True, delay=0.0, enable_eposide_timestep=True, noise_power=0):

    action_space_min = np.array(env.action_space.low)
    action_space_max = np.array(env.action_space.high)

    episode_timestep = 1
    done = True
    learn_count = 0
    save_count = 0

    for timestep in range(1, max_timestep + 1):

        episode_timestep += 1

        if done:
            episode_timestep = 1
            episode_cumulative_reward = 0
            next_observation_partial = env.reset()
            next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if enable_eposide_timestep else next_observation_partial

            if render: env.render()
            if render: time.sleep(delay)

        observation = next_observation

        action = agent.act(observation)
        limited_action = np.clip(action, -1, 1)
        corrupted_action = np.clip(limited_action + noise_power * np.random.randn(*limited_action.shape), -1, 1)
        scaled_action = (corrupted_action / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
        next_observation_partial, reward, done, info = env.step(scaled_action)
        next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if enable_eposide_timestep else next_observation_partial
        agent.record(observation, action, reward, next_observation, done)

        episode_cumulative_reward += reward
        if done: print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tTimestep: ' + str(
            timestep) + '\t\tTotal reward: ' + str(episode_cumulative_reward))

        if render: env.render()
        if render: time.sleep(delay)

        if done:
            if (timestep / learn_interval) >= (learn_count + 1):
                learn_count += 1
                agent.learn()
            if (timestep / save_interval) >= (save_count + 1):
                save_count += 1
                agent.save()

    agent.save()
    env.close()

print('Running...')

if args.time:
    pr = cProfile.Profile()
    pr.enable()

Loop(env, agent, max_timestep, learn_interval, save_interval, render=render, delay=delay, enable_eposide_timestep=enable_eposide_timestep, noise_power=noise_power)

if args.time:
    pr.disable()
    pr.dump_stats('profile.dat')

print('Finished')
print('')

if args.time:
    print('Follow the link below to see the time profile')
    os.system("snakeviz profile.dat")