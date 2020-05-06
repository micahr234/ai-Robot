import argparse
import cProfile
import os
import time
import numpy as np
from pathlib import Path

import gym
import pybullet
import pybullet_envs
import pybullet_data

from agentContinuous import AgentContinuous

parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', help='Instance name for this run - leave blank for new random name',  type=str, default='', required=False)
parser.add_argument('-e', '--environment', help='Name of gym environment', type=str, required=True)
parser.add_argument('-a', '--agent', help='Select RL agent to use', type=str, required=True)
parser.add_argument('-t', '--time', help='Time/profile code', action='store_true', required=False)
parser.add_argument('-r', '--render', help='Render environment', action='store_true', required=False)
parser.add_argument('-rd', '--render_delay', help='Time to wait in between action updates', type=float, default=0.0, required=False)
parser.add_argument('-v', '--verbosity', help='Log verbosity in tensorboard', action='store_true', required=False)

parser.add_argument('-mt', '--max_timestep', help='Max number of environment step', type=int, default=100000, required=False)
parser.add_argument('-li', '--learn_interval', help='Frequency at which learning takes place', type=int, default=1000, required=False)
parser.add_argument('-s', '--save', help='Save networks and experience replay memory', action='store_true', required=False)
parser.add_argument('-et', '--episode_timestamp', help='Include episode timestamp in state vector', action='store_true', required=False)
parser.add_argument('-an', '--action_noise', help='Adds gaussian noise to action', type=float, default=0.0, required=False)

parser.add_argument('-b', '--batches', help='Number of batches', type=int, default=100, required=False)
parser.add_argument('-bs', '--batch_size', help='Size of mini-batches', type=int, default=500, required=False)
parser.add_argument('-mbs', '--memory_buffer_size', help='Size of experience memory buffer', type=int, default=100000, required=False)
parser.add_argument('-d', '--discount', help='Discount factor', type=float, default=1.0, required=False)
parser.add_argument('-vlr', '--value_learn_rate', help='Value learn rate', type=float, default=0.001, required=False)
parser.add_argument('-plr', '--policy_learn_rate', help='Policy learn rate', type=float, default=0.001, required=False)
parser.add_argument('-prelr', '--preprocess_learn_rate', help='Preprocess learn rate', type=float, default=0.001, required=False)
parser.add_argument('-pd', '--policy_delay', help='Number of value learn iterations that take place for every policy learn iteration', type=int, default=10, required=False)
parser.add_argument('-nlf', '--next_learn_factor', help='Disbelief dampening of network', type=float, default=0.0, required=False)
parser.add_argument('-ar', '--action_randomness', help='Adds gaussian noise to action choice', type=float, default=0.0, required=False)
parser.add_argument('-vhls', '--value_hidden_layer_sizes', help='Hidden layer sizes in the value network', nargs='*', type=int, default=[256, 128], required=False)
parser.add_argument('-phls', '--policy_hidden_layer_sizes', help='Hidden layer sizes in the policy network', nargs='*', type=int, default=[256, 128], required=False)
parser.add_argument('-prehls', '--preprocess_hidden_layer_sizes', help='Hidden layer sizes in the preprocess network', nargs='*', type=int, default=[20, 10], required=False)

args = parser.parse_args()

if args.name == '':
    args.name = args.agent + args.environment + str(time.time())
print('Run name: ' + args.name)
print('')

print('Building environment: ' + args.environment)
if args.environment == 'CartPoleBulletEnv-v1':
    env = gym.make('CartPoleBulletEnv-v1', renders=args.render, discrete_actions=False)
    num_of_actions = 1
    num_of_states = 4 + (1 if args.episode_timestamp else 0)
else:
    env = gym.make(args.environment)
    num_of_actions = 0 #needs debug
    num_of_states = 0 + (1 if args.episode_timestamp else 0) #needs debug
print('')

print('Creating agent: ' + args.agent)
if args.agent == 'continuous':
    agent = AgentContinuous(args.name,  num_of_actions, num_of_states,
    value_hidden_layer_sizes = args.value_hidden_layer_sizes, policy_hidden_layer_sizes=args.policy_hidden_layer_sizes, preprocess_hidden_layer_sizes=args.preprocess_hidden_layer_sizes,
    batch_size=args.batch_size, learn_iterations=args.batches, memory_buffer_size=args.memory_buffer_size,
    discount=args.discount, value_learn_rate=args.value_learn_rate, policy_learn_rate=args.policy_learn_rate, preprocess_learn_rate=args.preprocess_learn_rate, policy_delay=args.policy_delay,
    next_learn_factor=args.next_learn_factor, randomness=args.action_randomness,
    verbosity=args.verbosity)
print('')


def Loop():

    action_space_min = np.array(env.action_space.low)
    action_space_max = np.array(env.action_space.high)

    episode_timestep = 1
    done = True
    learn_count = 0

    for timestep in range(1, args.max_timestep + 1):

        episode_timestep += 1

        if done:
            episode_timestep = 1
            episode_cumulative_reward = 0
            next_observation_partial = env.reset()
            next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if args.episode_timestamp else next_observation_partial

            if args.render: env.render()
            if args.render: time.sleep(args.delay)

        observation = next_observation

        action = agent.act(observation)
        corrupted_action = np.clip(action + args.action_noise * np.random.randn(*action.shape), -1, 1)
        scaled_action = (corrupted_action / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
        next_observation_partial, reward, done, info = env.step(scaled_action)
        next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if args.episode_timestamp else next_observation_partial
        agent.record(observation, action, reward, next_observation, done)

        episode_cumulative_reward += reward
        if done: print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tTimestep: ' + str(
            timestep) + '\t\tTotal reward: ' + str(episode_cumulative_reward))

        if args.render: env.render()
        if args.render: time.sleep(args.delay)

        if done:
            if (timestep / args.learn_interval) >= (learn_count + 1):
                learn_count += 1
                agent.learn()
            if args.save:
                agent.save()

    if args.save:
        agent.save()
    env.close()

print('Running...')

if args.time:
    pr = cProfile.Profile()
    pr.enable()

Loop()

if args.time:
    pr.disable()
    profile_dir = Path.cwd() / 'profile' / args.name
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    profile_filename = profile_dir / 'profile.pt'
    pr.dump_stats(profile_filename)

print('Finished')
print('')

if args.time:
    print('Follow the link below to see the time profile')
    os.system("snakeviz " + str(profile_filename))