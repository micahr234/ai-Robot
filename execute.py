import cProfile
import os
import time
import copy
import numpy as np
from pathlib import Path
from collections import deque

import gym
import pybullet
import pybullet_envs
import pybullet_data
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

import torch

from agent import Agent

def Execute(**params):

    log_params = {}

    # Error if any of these variables do not exist
    assert 'environment_name' in params, '"environment_name" variable required'
    environment_name = params['environment_name']
    log_params['environment_name'] = environment_name
    assert 'environment' in params, '"environment" variable required'
    environment = params['environment']
    assert 'previous_states' in params, '"previous_states" variable required'
    previous_states = params['previous_states']
    log_params['previous_states'] = previous_states
    assert 'agent_name' in params, '"agent_name" variable required'
    agent_name = params['agent_name']
    log_params['agent_name'] = agent_name

    assert 'num_of_latent_states' in params, '"num_of_latent_states" variable required'
    num_of_latent_states = params['num_of_latent_states']
    log_params['num_of_latent_states'] = num_of_latent_states
    assert 'num_of_random_states' in params, '"num_of_random_states" variable required'
    num_of_random_states = params['num_of_random_states']
    log_params['num_of_random_states'] = num_of_random_states

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

    assert 'state_input_transform' in params, '"state_input_transform" variable required'
    state_input_transform = params['state_input_transform']
    assert 'reward_input_transform' in params, '"reward_input_transform" variable required'
    reward_input_transform = params['reward_input_transform']
    assert 'action_input_transform' in params, '"action_input_transform" variable required'
    action_input_transform = params['action_input_transform']
    assert 'done_input_transform' in params, '"done_input_transform" variable required'
    done_input_transform = params['done_input_transform']
    assert 'action_output_transform' in params, '"action_output_transform" variable required'
    action_output_transform = params['action_output_transform']

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
    policy_learn_beta = params['policy_learn_beta'] if 'policy_learn_beta' in params else 0.0
    log_params['policy_learn_beta'] = policy_learn_beta
    batches = params['batches'] if 'batches' in params else 200
    log_params['batches'] = batches
    batch_size = params['batch_size'] if 'batch_size' in params else 2000
    log_params['batch_size'] = batch_size
    memory_buffer_size = params['memory_buffer_size'] if 'memory_buffer_size' in params else 10000
    log_params['memory_buffer_size'] = memory_buffer_size

    preprocess_learn_rate = params['preprocess_learn_rate'] if 'preprocess_learn_rate' in params else lambda batch: 0.0001
    log_params['preprocess_learn_rate'] = preprocess_learn_rate
    preprocess_latent_learn_factor = params['preprocess_latent_learn_factor'] if 'preprocess_latent_learn_factor' in params else lambda batch: 1.0
    log_params['preprocess_latent_learn_factor'] = preprocess_latent_learn_factor
    policy_value_learn_rate = params['policy_value_learn_rate'] if 'policy_value_learn_rate' in params else lambda batch: 0.0001
    log_params['policy_value_learn_rate'] = policy_value_learn_rate
    policy_entropy_learn_factor = params['policy_entropy_learn_factor'] if 'policy_entropy_learn_factor' in params else lambda batch: 0.0001
    log_params['policy_entropy_learn_factor'] = policy_entropy_learn_factor
    value_learn_rate = params['value_learn_rate'] if 'value_learn_rate' in params else lambda batch: 0.0001
    log_params['value_learn_rate'] = value_learn_rate
    value_next_learn_factor = params['value_next_learn_factor'] if 'value_next_learn_factor' in params else lambda batch: 0.8
    log_params['value_next_learn_factor'] = value_next_learn_factor
    discount = params['discount'] if 'discount' in params else lambda batch: 0.99
    log_params['discount'] = discount

    print('')

    print('Instance name: ' + instance_name)
    print('')

    print('Creating Tensor Board Log')
    tensor_board_dir = Path.cwd() / 'runs' / instance_name / str(time.time())
    tensor_board = torch.utils.tensorboard.SummaryWriter(tensor_board_dir, max_queue=10000, flush_secs=60)
    tensor_board.add_text('Hyper Params', str(log_params), 0)
    print('')

    #if environment_name == 'CartPoleBulletEnv-v1':
    #    env = gym.make(environment_name, renders=render, discrete_actions=False)
    #elif environment_name == 'HopperBulletEnv-v0':
    #    env = gym.make(environment_name, render=render)
    #else:
    #    env = gym.make(environment_name)

    print('Creating agent: ' + agent_name)
    if agent_name == 'agent':
        agent = Agent(
            instance_name,
            num_of_latent_states,
            num_of_random_states,
            preprocess_fwd_net,
            preprocess_rev_net,
            value_net,
            policy_net,
            tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            preprocess_learn_rate=preprocess_learn_rate,
            preprocess_latent_learn_factor=preprocess_latent_learn_factor,
            policy_value_learn_rate=policy_value_learn_rate,
            policy_entropy_learn_factor=policy_entropy_learn_factor,
            value_learn_rate=value_learn_rate,
            value_next_learn_factor=value_next_learn_factor,
            discount=discount,
            verbosity=verbosity,
            state_input_transform=state_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            done_input_transform=done_input_transform,
            action_output_transform=action_output_transform
        )
    else:
        raise ValueError('Agent name in not valid')
    print('')
    
    def Record(tensor_board, timestep, state, action, reward, done, episode_timestep, episode_cumulative_reward):
        
        #if verbosity:
        #    for n in range(self.num_of_states):
        #        self.tensor_board.add_scalar('Record/state' + str(n), in_state[n], timestep)
        #        self.tensor_board.add_scalar('Record/next_state' + str(n), in_next_state[n], timestep)
        #    for n in range(self.num_of_actions):
        #        self.tensor_board.add_scalar('Record/action' + str(n), in_action[n], timestep)
        #    self.tensor_board.add_scalar('Record/reward', in_reward, timestep)
        #    self.tensor_board.add_scalar('Record/in_done', in_done, timestep)
        
        if done:
            print('Episode finished\t\tEpisode timestep: ' + str(episode_timestep) + '\t\tCumulative reward: ' + str(episode_cumulative_reward))
            tensor_board.add_scalar('Record/cumulative_reward', episode_cumulative_reward, timestep)

    def Loop():

        action_space_min = np.array(environment.action_space.low)
        action_space_max = np.array(environment.action_space.high)

        done = True
        learnstep = 1

        state_buffer = deque(maxlen=previous_states)
        next_state_buffer = deque(maxlen=previous_states)

        for timestep in range(1, max_timestep + 1):

            if done:
                episode_timestep = 1
                episode_cumulative_reward = 0
                next_state_partial = environment.reset()
                next_state = np.concatenate((next_state_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_state_partial
                next_state_buffer.clear()
                for i in range(0, previous_states):
                    next_state_buffer.append(next_state)

                if render: environment.render()
                if render: time.sleep(render_delay)

            state = next_state
            state_buffer = copy.deepcopy(next_state_buffer)

            action = agent.act(state_buffer)
            #action = agent.act(state_buffer[-1])
            #action = agent.act(state)
            scaled_action = (np.array(action) / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
            next_state_partial, reward, done, info = environment.step(scaled_action)
            next_state = np.concatenate((next_state_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_state_partial
            next_state_buffer.append(next_state)
            agent.record(state_buffer, action, reward, next_state_buffer, done)
            #agent.record(state_buffer[-1], action, reward, next_state_buffer[-1], done)
            #agent.record(state, action, reward, next_state, done)

            episode_cumulative_reward += reward
            Record(tensor_board, timestep, state, action, reward, done, episode_timestep, episode_cumulative_reward)

            if render: environment.render()
            if render: time.sleep(render_delay)

            if (timestep / learn_interval) >= learnstep:
                learnstep += 1
                agent.learn()
                if save:
                    agent.save()

            episode_timestep += 1

        if save:
            agent.save()
        environment.close()

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