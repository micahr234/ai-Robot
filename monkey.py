import torch
import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from agents.model_free_deterministic_agent import model_free_deterministic_agent
from lib.tensor_board import tensor_board
from lib.game import game

num_of_observations = 4
num_of_actions = 2

latent_net = torch.nn.Sequential(
)

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_observations, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions)
)

value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_observations + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

def scale(tensor, input_min, input_max):
    slope = (input_max - input_min) * 0.5
    intercept = (input_max + input_min) * 0.5
    scaled_tensor = (tensor - intercept) / slope
    return scaled_tensor

def observation_input_transform(state):
    state_direct_xform = torch.FloatTensor(state).contiguous()
    state_indirect_xform = torch.FloatTensor([])
    state_indirect_target_xform = torch.FloatTensor([])
    return state_direct_xform, state_indirect_xform, state_indirect_target_xform

def reward_input_transform(reward):
    reward_xform = torch.FloatTensor([reward]).contiguous()
    return reward_xform

def action_input_transform(action):
    action_xform = torch.FloatTensor(action).contiguous()
    return action_xform

def survive_input_transform(done):
    survive_xform = torch.FloatTensor([float(done)]).contiguous()
    return survive_xform

def action_output_transform(action):
    action_xform = action.tolist()
    return action_xform

name='Monkey1'

unity_env = UnityEnvironment('environments/Monkey/Monkey.exe', base_port=10000, worker_id=np.random.randint(1000), no_graphics=False, timeout_wait=None)
env_instance = UnityToGymWrapper(unity_env, False, False, True)

tensor_board_instance = tensor_board(name=name)

agent_instance = model_free_deterministic_agent(
    name=name,
    value_net=value_net,
    policy_net=policy_net,
    tensor_board=tensor_board_instance,
    action_random_prob=lambda i: 0.3,
    value_learn_rate=lambda i: 0.0001,
    value_discount=0.99,
    value_polyak=0.001,
    policy_learn_rate=lambda i: 0.0001,
    policy_polyak=0.001,
    gpu=0)

game_instance = game(
    name=name,
    environment=env_instance,
    agent=agent_instance,
    tensor_board=tensor_board_instance,
    learn_interval=100,
    max_timestep=40000,
    memory_buffer_size=40000,
    max_episode_timestamp=None,
    batch_size=200,
    batches=100,
    render_delay=0,
    save=False,
    profile=False,
    observation_input_transform=observation_input_transform,
    action_input_transform=action_input_transform,
    reward_input_transform=reward_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
)

game_instance.run()

env_instance.close()