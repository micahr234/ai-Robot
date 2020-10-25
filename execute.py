import cProfile
import os
import time
import copy
import numpy as np
from pathlib import Path
from collections import deque

import torch

import agent_model_based_stochastic_actor
import agent_model_based_deterministic_actor
#import agent_model_free_stochastic_actor
#import agent_model_free_deterministic_actor
import agent_image_test

import game


def execute(
        instance_name=None,

        environment_name=None,
        environment=None,
        observation_input_transform=None,
        reward_input_transform=None,
        action_input_transform=None,
        survive_input_transform=None,
        action_output_transform=None,
        max_episode_timestamp=None,
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
        policy_mix_net=None,
        state_frames=None,
        latent_states=None,
        action_distributions=None,
        action_random_prob=None,

        latent_learn_rate=None,
        latent_polyak=None,
        model_learn_rate=None,
        reward_learn_rate=None,
        survive_learn_rate=None,
        env_polyak=None,
        value_learn_rate=None,
        value_hallu_loops=None,
        value_discount=None,
        value_polyak=None,
        policy_learn_rate=None,
        policy_polyak=None,

        profile=None,
        log_level=None,
        gpu=None
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
        log_params['max_episode_timestamp'] = max_episode_timestamp

        log_params['latent_net'] = latent_net
        log_params['value_net'] = value_net
        log_params['policy_net'] = policy_net
        log_params['policy_mix_net'] = policy_mix_net
        log_params['model_net'] = policy_net
        log_params['reward_net'] = reward_net
        log_params['survive_net'] = survive_net
        log_params['state_frames'] = state_frames
        log_params['latent_states'] = latent_states
        log_params['action_distributions'] = action_distributions
        log_params['action_random_prob'] = action_random_prob

        log_params['latent_learn_rate'] = latent_learn_rate
        log_params['policy_polyak'] = policy_polyak
        log_params['model_learn_rate'] = model_learn_rate
        log_params['reward_learn_rate'] = reward_learn_rate
        log_params['survive_learn_rate'] = survive_learn_rate
        log_params['env_polyak'] = env_polyak
        log_params['value_learn_rate'] = value_learn_rate
        log_params['value_hallu_loops'] = value_hallu_loops
        log_params['value_discount'] = value_discount
        log_params['value_polyak'] = value_polyak
        log_params['policy_learn_rate'] = policy_learn_rate
        log_params['policy_polyak'] = policy_polyak

        tensor_board.add_text('Hyper Params', str(log_params), 0)

    print('')

    # if environment_name == 'CartPoleBulletEnv-v1':
    #    env = gym.make(environment_name, renders=render, discrete_actions=False)
    # elif environment_name == 'HopperBulletEnv-v0':
    #    env = gym.make(environment_name, render=render)
    # else:
    #    env = gym.make(environment_name)

    print('Creating agent: ' + agent_name)
    if agent_name == 'agent_model_based_stochastic_actor':
        agent = agent_model_based_stochastic_actor.agent(
            name=instance_name,
            latent_states=latent_states,
            action_distributions=action_distributions,
            latent_net=latent_net,
            value_net=value_net,
            policy_net=policy_net,
            policy_mix_net=policy_mix_net,
            model_net=model_net,
            reward_net=reward_net,
            survive_net=survive_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            latent_learn_rate=latent_learn_rate,
            latent_polyak=latent_polyak,
            model_learn_rate=model_learn_rate,
            reward_learn_rate=reward_learn_rate,
            survive_learn_rate=survive_learn_rate,
            env_polyak=env_polyak,
            value_learn_rate=value_learn_rate,
            value_hallu_loops=value_hallu_loops,
            value_polyak=value_polyak,
            policy_learn_rate=policy_learn_rate,
            policy_polyak=policy_polyak,
            log_level=log_level,
            gpu=gpu
        )
    elif agent_name == 'agent_model_based_deterministic_actor':
        agent = agent_model_based_deterministic_actor.agent(
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
            action_random_prob=action_random_prob,
            latent_learn_rate=latent_learn_rate,
            model_learn_rate=model_learn_rate,
            reward_learn_rate=reward_learn_rate,
            survive_learn_rate=survive_learn_rate,
            value_learn_rate=value_learn_rate,
            value_hallu_loops=value_hallu_loops,
            value_polyak=value_polyak,
            policy_learn_rate=policy_learn_rate,
            policy_polyak=policy_polyak,
            log_level=log_level,
            gpu=gpu
        )
    elif agent_name == 'agent_model_free_stochastic_actor':
        agent = agent_model_free_stochastic_actor.agent(
            name=instance_name,
            action_distributions=action_distributions,
            value_net=value_net,
            policy_net=policy_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            value_learn_rate=value_learn_rate,
            value_action_samples=value_action_samples,
            value_discount=value_discount,
            policy_learn_rate=policy_learn_rate,
            policy_action_samples=policy_action_samples,
            log_level=log_level,
            state_input_transform=observation_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            survive_input_transform=survive_input_transform,
            action_output_transform=action_output_transform,
            gpu=gpu
        )
    elif agent_name == 'agent_model_free_deterministic_actor':
        agent = agent_model_free_deterministic_actor.agent(
            name=instance_name,
            value_net=value_net,
            policy_net=policy_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            action_random_prob=action_random_prob,
            value_learn_rate=value_learn_rate,
            value_discount=value_discount,
            policy_learn_rate=policy_learn_rate,
            log_level=log_level,
            state_input_transform=observation_input_transform,
            reward_input_transform=reward_input_transform,
            action_input_transform=action_input_transform,
            survive_input_transform=survive_input_transform,
            action_output_transform=action_output_transform,
            gpu=gpu
        )
    elif agent_name == 'agent_image_test':
        agent = agent_image_test.agent(
            name=instance_name,
            latent_states=latent_states,
            latent_net=latent_net,
            tensor_board=tensor_board,
            batch_size=batch_size,
            batches=batches,
            memory_buffer_size=memory_buffer_size,
            latent_learn_rate=latent_learn_rate,
            log_level=log_level,
            gpu=gpu
        )
    else:
        raise ValueError('Agent name in not valid')
    print('')

    print('Running...')

    game_instance = game.game(
        instance_name,
        environment,
        agent,
        tensor_board,
        log_level,
        state_frames,
        learn_interval,
        max_timestep,
        memory_buffer_size,
        max_episode_timestamp,
        render_delay,
        save,
        observation_input_transform,
        action_input_transform,
        reward_input_transform,
        survive_input_transform,
        action_output_transform,
    )

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    game_instance.run()

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
