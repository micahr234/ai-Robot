import torch
import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from agents.model_free_deterministic_agent import model_free_deterministic_agent
from lib.tensor_board import tensor_board
from lib.game import game

num_of_observations = 5
num_of_actions = 2

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

def observation_input_transform(state, timestep):
    state_direct_xform_pre = torch.FloatTensor(state).contiguous()
    state_indirect_xform = torch.FloatTensor([])
    state_indirect_target_xform = torch.FloatTensor([])
    max_episode_timestamp = 100
    scaled_timestep = (torch.FloatTensor([[timestep]]) / max_episode_timestamp) * 2.0 - 1.0
    state_direct_xform = torch.cat((state_direct_xform_pre, scaled_timestep), dim=-1)
    return state_direct_xform, state_indirect_xform, state_indirect_target_xform

def reward_input_transform(reward):
    reward_xform = torch.FloatTensor([reward]).contiguous()
    return reward_xform

def survive_input_transform(done):
    survive_xform = torch.FloatTensor([float(done)]).contiguous()
    return survive_xform

def action_output_transform(action):
    action_xform = action.tolist()
    return action_xform

name='Monkey1'


num_of_envs = 5
env_instance = []
for env_num in range(num_of_envs):
    no_graphics = False
    unity_env = UnityEnvironment('environments/Monkey/Monkey.exe', base_port=10000+int(torch.randint(10000, (1, ))),
                                 worker_id=env_num, no_graphics=no_graphics, timeout_wait=None)
    env_instance.append(UnityToGymWrapper(unity_env, False, False, True))

action_random_prob = torch.linspace(0, 0.5, num_of_envs)

tensor_board_instance = tensor_board(name=name)

agent_instance = model_free_deterministic_agent(
    name=name,
    value_net=value_net,
    policy_net=policy_net,
    value_learn_rate=0.0001,
    value_discount=0.99,
    value_polyak=0.001,
    policy_learn_rate=0.0001,
    policy_polyak=0.001)

game_instance = game(
    name=name,
    environment=env_instance,
    agent=agent_instance,
    tensor_board=tensor_board_instance,
    steps=100,
    epochs=10000,
    memory_buffer_size=100000*num_of_envs,
    batch_size=200,
    save=False,
    profile=False,
    observation_input_transform=observation_input_transform,
    reward_input_transform=reward_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    action_random_prob=action_random_prob)

game_instance.run()

for e in env_instance:
    e.close()