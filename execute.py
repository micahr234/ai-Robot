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

def Execute(
            instance_name,

            environment_name,
            environment,
            state_input_transform,
            reward_input_transform,
            action_input_transform,
            terminate_input_transform,
            action_output_transform,
            episode_timestamp,
            render,
            render_delay,

            agent_name,
            max_timestep,
            learn_interval,
            batches,
            batch_size,
            memory_buffer_size,
            save,

            latent_fwd_net,
            latent_rev_net,
            model_net,
            reward_net,
            terminate_net,
            value_net,
            policy_net,
            previous_states,
            latent_states,

            latent_learn_rate,
            latent_latent_learn_factor,
            model_learn_rate,
            reward_learn_rate,
            terminate_learn_rate,
            value_learn_rate,
            value_next_learn_factor,
            discount,
            policy_learn_rate,

            profile,
            verbosity,
            ):

    print('')

    instance_name = environment_name + agent_name + str(time.time()) if instance_name is None else instance_name
    print('Instance name: ' + instance_name)
    print('')

    print('Creating Tensor Board Log')
    log_params = {}
    log_params['environment_name'] = environment_name
    log_params['previous_states'] = previous_states
    log_params['agent_name'] = agent_name
    log_params['latent_states'] = latent_states
    log_params['latent_fwd_net'] = latent_fwd_net
    log_params['latent_rev_net'] = latent_rev_net
    log_params['value_net'] = value_net
    log_params['policy_net'] = policy_net
    log_params['model_net'] = policy_net
    log_params['reward_net'] = reward_net
    log_params['terminate_net'] = terminate_net
    log_params['instance_name'] = instance_name
    log_params['max_timestep'] = max_timestep
    log_params['learn_interval'] = learn_interval
    log_params['episode_timestamp'] = episode_timestamp
    log_params['batches'] = batches
    log_params['batch_size'] = batch_size
    log_params['memory_buffer_size'] = memory_buffer_size
    log_params['latent_learn_rate'] = latent_learn_rate
    log_params['latent_latent_learn_factor'] = latent_latent_learn_factor
    log_params['policy_learn_rate'] = policy_learn_rate
    log_params['value_learn_rate'] = value_learn_rate
    log_params['value_next_learn_factor'] = value_next_learn_factor
    log_params['discount'] = discount
    log_params['model_learn_rate'] = model_learn_rate
    log_params['reward_learn_rate'] = reward_learn_rate
    log_params['terminate_learn_rate'] = terminate_learn_rate
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
            name=instance_name,
            latent_states=latent_states,
            latent_fwd_net=latent_fwd_net,
            latent_rev_net=latent_rev_net,
            value_net=value_net,
            policy_net=policy_net,
            model_net=model_net,
            reward_net=reward_net,
            terminate_net=terminate_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            latent_learn_rate=latent_learn_rate,
            latent_latent_learn_factor=latent_latent_learn_factor,
            policy_learn_rate=policy_learn_rate,
            value_learn_rate=value_learn_rate,
            value_next_learn_factor=value_next_learn_factor,
            discount=discount,
            model_learn_rate=model_learn_rate,
            reward_learn_rate=reward_learn_rate,
            terminate_learn_rate=terminate_learn_rate,
            verbosity=verbosity,
            state_input_transform=state_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            terminate_input_transform=terminate_input_transform,
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

        observation_buffer = deque(maxlen=previous_states)
        observation_next_buffer = deque(maxlen=previous_states)

        for timestep in range(1, max_timestep + 1):

            if done:
                episode_timestep = 1
                episode_cumulative_reward = 0
                next_state_partial = environment.reset()
                observation_next = np.concatenate((next_state_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_state_partial
                observation_next_buffer.clear()
                for i in range(0, previous_states):
                    observation_next_buffer.append(observation_next)

                if render: environment.render()
                if render: time.sleep(render_delay)

            observation = observation_next
            observation_buffer = copy.deepcopy(observation_next_buffer)

            action = agent.act(observation_buffer)
            scaled_action = (np.array(action) / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
            next_state_partial, reward, done, info = environment.step(scaled_action)
            observation_next = np.concatenate((next_state_partial, np.array(episode_timestep, ndmin=1))) if episode_timestamp else next_state_partial
            observation_next_buffer.append(observation_next)
            agent.record(observation_buffer, action, reward, observation_next_buffer, done)

            episode_cumulative_reward += reward
            Record(tensor_board, timestep, observation, action, reward, done, episode_timestep, episode_cumulative_reward)

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
