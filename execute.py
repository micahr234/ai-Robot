import cProfile
import os
import time
import copy
import numpy as np
from pathlib import Path
from collections import deque

import torch

from agent_model_based_stochastic_actor import agent_model_based_stochastic_actor
from agent_model_free_stochastic_actor import agent_model_free_stochastic_actor

def Execute(
            instance_name=None,

            environment_name=None,
            environment=None,
            state_input_transform=None,
            reward_input_transform=None,
            action_input_transform=None,
            survive_input_transform=None,
            action_output_transform=None,
            episode_timestamp=None,
            render=None,
            render_delay=None,

            agent_name=None,
            max_timestep=None,
            learn_interval=None,
            batches=None,
            batch_size=None,
            memory_buffer_size=None,
            save=None,

            latent_net=None,
            model_net=None,
            reward_net=None,
            survive_net=None,
            value_net=None,
            policy_net=None,
            state_frames=None,
            latent_states=None,

            latent_learn_rate=None,
            model_learn_rate=None,
            reward_learn_rate=None,
            survive_learn_rate=None,
            value_learn_rate=None,
            value_next_learn_factor=None,
            value_action_samples=None,
            value_hallu_loops=None,
            policy_learn_rate=None,
            policy_action_samples=None,

            profile=None,
            log_level=None
            ):

    print('')

    instance_name = environment_name + agent_name + str(time.time()) if instance_name is None else instance_name
    print('Instance name: ' + instance_name)
    print('')

    print('Creating Tensor Board Log')
    tensor_board_dir = Path.cwd() / 'runs' / instance_name / str(time.time())
    tensor_board = torch.utils.tensorboard.SummaryWriter(tensor_board_dir, max_queue=10000, flush_secs=60)
    # log_level 0=None, 1=Hparams & Reward & Loss, 2=Everything in 2 & Learning Rates, 3=Everything in 2 & Distributions

    if log_level >= 1:
        log_params = {}
        log_params['environment_name'] = environment_name
        log_params['agent_name'] = agent_name
        log_params['instance_name'] = instance_name

        log_params['learn_interval'] = learn_interval
        log_params['batches'] = batches
        log_params['batch_size'] = batch_size
        log_params['memory_buffer_size'] = memory_buffer_size

        log_params['latent_net'] = latent_net
        log_params['value_net'] = value_net
        log_params['policy_net'] = policy_net
        log_params['model_net'] = policy_net
        log_params['reward_net'] = reward_net
        log_params['survive_net'] = survive_net
        log_params['state_frames'] = state_frames
        log_params['latent_states'] = latent_states

        log_params['latent_learn_rate'] = latent_learn_rate
        log_params['model_learn_rate'] = model_learn_rate
        log_params['reward_learn_rate'] = reward_learn_rate
        log_params['survive_learn_rate'] = survive_learn_rate
        log_params['value_learn_rate'] = value_learn_rate
        log_params['value_next_learn_factor'] = value_next_learn_factor
        log_params['value_action_samples'] = value_action_samples
        log_params['value_hallu_loops'] = value_hallu_loops
        log_params['policy_learn_rate'] = policy_learn_rate
        log_params['policy_action_samples'] = policy_action_samples

        tensor_board.add_text('Hyper Params', str(log_params), 0)

    print('')

    #if environment_name == 'CartPoleBulletEnv-v1':
    #    env = gym.make(environment_name, renders=render, discrete_actions=False)
    #elif environment_name == 'HopperBulletEnv-v0':
    #    env = gym.make(environment_name, render=render)
    #else:
    #    env = gym.make(environment_name)

    print('Creating agent: ' + agent_name)
    if agent_name == 'agent_model_based_stochastic_actor':
        agent = agent_model_based_stochastic_actor(
            name=instance_name,
            latent_states=latent_states,
            latent_net=latent_net,
            value_net=value_net,
            policy_net=policy_net,
            model_net=model_net,
            reward_net=reward_net,
            survive_net=survive_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            latent_learn_rate=latent_learn_rate,
            model_learn_rate=model_learn_rate,
            reward_learn_rate=reward_learn_rate,
            survive_learn_rate=survive_learn_rate,
            value_learn_rate=value_learn_rate,
            value_next_learn_factor=value_next_learn_factor,
            value_action_samples=value_action_samples,
            value_hallu_loops=value_hallu_loops,
            policy_learn_rate=policy_learn_rate,
            policy_action_samples=policy_action_samples,
            log_level=log_level,
            state_input_transform=state_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            survive_input_transform=survive_input_transform,
            action_output_transform=action_output_transform
        )
    elif agent_name == 'agent_model_free_stochastic_actor':
        agent = agent_model_free_stochastic_actor(
            name=instance_name,
            value_net=value_net,
            policy_net=policy_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            value_learn_rate=value_learn_rate,
            value_next_learn_factor=value_next_learn_factor,
            value_action_samples=value_action_samples,
            policy_learn_rate=policy_learn_rate,
            policy_action_samples=policy_action_samples,
            log_level=log_level,
            state_input_transform=state_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            survive_input_transform=survive_input_transform,
            action_output_transform=action_output_transform
        )
    else:
        raise ValueError('Agent name in not valid')
    print('')


    def log_reward(tensor_board, timestep, survive, episode_cumulative_reward):

        if log_level >= 1:
            if survive == 0.0:
                tensor_board.add_scalar('record/cumulative_reward', episode_cumulative_reward, timestep)

    log_observation_buffer = deque()
    log_action_buffer = deque()
    log_reward_buffer = deque()
    log_survive_buffer = deque()

    def log_values(tensor_board, timestep, observation, action, reward, survive, exe):

        if log_level >= 3:

            log_observation_buffer.append(observation)
            log_action_buffer.append(action)
            log_reward_buffer.append(reward)
            log_survive_buffer.append(survive)

            if exe:

                log_observation_tensor = torch.Tensor(log_observation_buffer)
                for n in range(log_observation_tensor.shape[1]):
                    tensor_board.add_histogram('observation_param' + str(n), log_observation_tensor[:, n], timestep)
                log_action_tensor = torch.Tensor(log_action_buffer)
                for n in range(log_action_tensor.shape[1]):
                    tensor_board.add_histogram('action_param' + str(n), log_action_tensor[:, n], timestep)
                tensor_board.add_histogram('reward', torch.Tensor(log_reward_buffer), timestep)
                tensor_board.add_histogram('survive', torch.Tensor(log_survive_buffer), timestep)

                log_observation_buffer.clear()
                log_action_buffer.clear()
                log_reward_buffer.clear()
                log_survive_buffer.clear()

    def Loop():

        action_space_min = np.array(environment.action_space.low)
        action_space_max = np.array(environment.action_space.high)

        survive = 0.0
        learnstep = 1

        observation_buffer = deque(maxlen=state_frames)
        observation_next_buffer = deque(maxlen=state_frames)

        for timestep in range(1, max_timestep + 1):

            learn = (timestep / learn_interval) >= learnstep

            if survive == 0.0:
                episode_timestep = 1
                episode_cumulative_reward = 0
                next_observation_partial = environment.reset()
                if episode_timestamp:
                    observation_next = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
                else:
                    observation_next = next_observation_partial
                observation_next_buffer.clear()
                for i in range(0, state_frames):
                    observation_next_buffer.append(observation_next)

                if render: environment.render()
                if render: time.sleep(render_delay)

            observation = observation_next
            observation_buffer = copy.deepcopy(observation_next_buffer)

            action = agent.act(observation_buffer)

            if not np.any(np.logical_and(-1.0 <= np.array(action), np.array(action) <= 1.0)):
                raise ValueError('Action cannot be outside range -1 to 1.')

            scaled_action = (np.array(action) / 2 + 0.5) * (action_space_max - action_space_min) + action_space_min
            next_observation_partial, reward, terminate, info = environment.step(scaled_action)
            survive = 1.0 - terminate
            if episode_timestamp:
                observation_next = np.concatenate((next_observation_partial, np.array(episode_timestep, ndmin=1)))
            else:
                observation_next = next_observation_partial
            observation_next_buffer.append(observation_next)
            agent.record(observation_buffer, action, reward, observation_next_buffer, survive)

            episode_cumulative_reward += reward
            log_reward(tensor_board, timestep, survive, episode_cumulative_reward)
            log_values(tensor_board, timestep, observation, action, reward, survive, learn)
            if survive == 0.0:
                print('---Terminated episode---\t\t\t\tEpisode timestep: ' + str(episode_timestep) + '\t\t\t\tCumulative reward: ' + str(episode_cumulative_reward))

            if render: environment.render()
            if render: time.sleep(render_delay)

            if learn:
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
