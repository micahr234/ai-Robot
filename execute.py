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
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import Agent

def Execute(**params):

    log_params = {}

    # Error if any of these variables do not exist
    assert 'environment_name' in params, '"environment_name" variable required'
    environment_name = params['environment_name']
    log_params['environment_name'] = environment_name
    assert 'agent_name' in params, '"agent_name" variable required'
    agent_name = params['agent_name']
    log_params['agent_name'] = agent_name

    assert 'num_of_actions' in params, '"num_of_actions" variable required'
    num_of_actions = params['num_of_actions']
    log_params['num_of_actions'] = num_of_actions
    assert 'num_of_states' in params, '"num_of_states" variable required'
    num_of_states = params['num_of_states']
    log_params['num_of_states'] = num_of_states
    assert 'num_of_latent_states' in params, '"num_of_latent_states" variable required'
    num_of_latent_states = params['num_of_latent_states']
    log_params['num_of_latent_states'] = num_of_latent_states

    assert 'preprocess_fwd_net' in params, '"preprocess_fwd_net" variable required'
    preprocess_fwd_net = params['preprocess_fwd_net']
    log_params['preprocess_fwd_net'] = preprocess_fwd_net
    assert 'preprocess_rev_net' in params, '"preprocess_rev_net" variable required'
    preprocess_rev_net = params['preprocess_rev_net']
    log_params['preprocess_rev_net'] = preprocess_rev_net
    assert 'value_net' in params, '"value_net" variable required'
    value_net = params['value_net']
    log_params['value_net'] = value_net
    assert 'policy_net' in params, '"policy_net" variable required'
    policy_net = params['policy_net']
    log_params['policy_net'] = policy_net

    # Assign these variables default values
    instance_name = params['instance_name'] if 'instance_name' in params else environment_name + agent_name + str(time.time())
    log_params['instance_name'] = instance_name
    profile = params['profile'] if 'profile' in params else False
    render = params['render'] if 'render' in params else False
    render_delay = params['render_delay'] if 'render_delay' in params else False
    verbosity = params['verbosity'] if 'verbosity' in params else False

    max_timestep = params['max_timestep'] if 'max_timestep' in params else 100000
    log_params['max_timestep'] = max_timestep
    learn_interval = params['learn_interval'] if 'learn_interval' in params else 2000
    log_params['learn_interval'] = learn_interval
    save = params['save'] if 'save' in params else False
    episode_timestamp = params['episode_timestamp'] if 'episode_timestamp' in params else True
    log_params['episode_timestamp'] = episode_timestamp
    action_randomness = params['action_randomness'] if 'action_randomness' in params else 0.0
    log_params['action_randomness'] = action_randomness
    batches = params['batches'] if 'batches' in params else 200
    log_params['batches'] = batches
    batch_size = params['batch_size'] if 'batch_size' in params else 2000
    log_params['batch_size'] = batch_size
    memory_buffer_size = params['memory_buffer_size'] if 'memory_buffer_size' in params else 10000
    log_params['memory_buffer_size'] = memory_buffer_size
    value_learn_rate = params['value_learn_rate'] if 'value_learn_rate' in params else 0.001
    log_params['value_learn_rate'] = value_learn_rate
    policy_learn_rate = params['policy_learn_rate'] if 'policy_learn_rate' in params else 0.001
    log_params['policy_learn_rate'] = policy_learn_rate
    preprocess_learn_rate = params['preprocess_learn_rate'] if 'preprocess_learn_rate' in params else 0.001
    log_params['preprocess_learn_rate'] = preprocess_learn_rate
    discount = params['discount'] if 'discount' in params else 1.0
    log_params['discount'] = discount
    policy_delay = params['policy_delay'] if 'policy_delay' in params else 10
    log_params['policy_delay'] = policy_delay
    preprocess_learn_beta = params['preprocess_learn_beta'] if 'preprocess_learn_beta' in params else 1.0
    log_params['preprocess_learn_beta'] = preprocess_learn_beta
    next_learn_factor = params['next_learn_factor'] if 'next_learn_factor' in params else 0.8
    log_params['next_learn_factor'] = next_learn_factor

    print('')

    print('Instance name: ' + instance_name)
    print('')

    print('Creating Tensor Board Log')
    tensor_board_dir = Path.cwd() / 'runs' / instance_name / str(time.time())
    tensor_board = SummaryWriter(tensor_board_dir, max_queue=10000, flush_secs=60)
    tensor_board.add_text('Hyper Params', str(log_params), 0)
    print('')

    print('Building environment: ' + environment_name)
    if environment_name == 'CartPoleBulletEnv-v1':
        env = gym.make(environment_name, renders=render, discrete_actions=False)
    elif environment_name == 'HopperBulletEnv-v0':
        env = gym.make(environment_name, render=render)
    else:
        env = gym.make(environment_name)
    print('')

    print('Creating agent: ' + agent_name)
    if agent_name == 'agent':
        agent = Agent(
            agent_name,
            num_of_actions,
            num_of_states,
            num_of_latent_states,
            preprocess_fwd_net,
            preprocess_rev_net,
            value_net,
            policy_net,
            tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            value_learn_rate=value_learn_rate,
            policy_learn_rate=policy_learn_rate,
            preprocess_learn_rate=preprocess_learn_rate,
            discount=discount,
            policy_delay=policy_delay,
            preprocess_learn_beta=preprocess_learn_beta,
            next_learn_factor=next_learn_factor,
            action_randomness=action_randomness,
            verbosity=verbosity
        )
    else:
        raise ValueError('Agent name in not valid')
    print('')


    def Loop():

        action_space_min = np.array(env.action_space.low)
        action_space_max = np.array(env.action_space.high)

        episode_timestep = 1
        done = True
        learn_count = 0

        for timestep in range(1, max_timestep + 1):

            episode_timestep += 1

            if done:
                episode_timestep = 1
                episode_cumulative_reward = 0
                next_observation_partial = env.reset()
                next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_observation_partial

                if render: env.render()
                if render: time.sleep(render_delay)

            observation = next_observation

            action = agent.act(observation)
            scaled_action = (action / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
            next_observation_partial, reward, done, info = env.step(scaled_action)
            next_observation = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_observation_partial
            agent.record(observation, action, reward, next_observation, done)

            episode_cumulative_reward += reward
            if done: print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tTimestep: ' + str(
                timestep) + '\t\tTotal reward: ' + str(episode_cumulative_reward))

            if render: env.render()
            if render: time.sleep(render_delay)

            if done:
                if (timestep / learn_interval) >= (learn_count + 1):
                    learn_count += 1
                    agent.learn()
                if save:
                    agent.save()

        if save:
            agent.save()
        env.close()

    print('Running...')

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    Loop()

    if profile:
        pr.disable()
        profile_dir = Path.cwd() / 'profile' / instance_name
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        profile_filename = profile_dir / 'profile.pt'
        pr.dump_stats(profile_filename)

    print('Finished')
    print('')

    if profile:
        print('Follow the link below to see the time profile')
        os.system("snakeviz " + str(profile_filename))