import torch
import math
import pybullet
import pybullet_envs
import pybullet_data
import gym
from execute import Execute
import numpy as np

num_of_states = 5
state_frames = 1
num_of_actions = 1
action_distributions = 5

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states * state_frames, 512),
    torch.nn.ReLU(),
    #torch.nn.Linear(512, 256),
    #torch.nn.ELU(),
    #torch.nn.Linear(256, 128),
    #torch.nn.ELU(),
    torch.nn.Linear(512, num_of_actions * 3 * action_distributions)
)

value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states * state_frames + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

def scale(input, input_min, input_max):
    slope = (input_max - input_min) * 0.5
    intercept = (input_max + input_min) * 0.5
    output = (input - intercept) / slope
    return output

def state_input_transform(state):
    xform_state = torch.Tensor([state]).contiguous()
    xform_state = scale(xform_state, torch.tensor([-1.0] * (num_of_states - 1) + [0]), torch.tensor([1.0] * (num_of_states - 1) + [199]))
    return xform_state

def reward_input_transform(reward):
    xform_reward = torch.Tensor([[reward]]).contiguous()
    return xform_reward

def action_input_transform(action):
    xform_action = torch.Tensor([action]).contiguous()
    return xform_action

def survive_input_transform(done):
    xform_reward = torch.Tensor([[float(done)]]).contiguous()
    return xform_reward

def action_output_transform(action):
    xform_action = action.tolist()
    return xform_action

gym_env = gym.make('CartPoleBulletEnv-v1', renders=True, discrete_actions=False)

Execute(
    instance_name='CartPole1',

    environment_name='CartPole',
    environment=gym_env,
    state_input_transform=state_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    episode_timestamp=True,
    render=False,
    render_delay=0.0,

    agent_name='agent_model_free_stochastic_actor',
    max_timestep=20000,
    learn_interval=500,
    batches=500,
    batch_size=200,
    memory_buffer_size=20000,
    save=False,

    value_net=value_net,
    policy_net=policy_net,
    state_frames=state_frames,

    value_learn_rate=lambda batch: 0.001,
    value_next_learn_factor=lambda batch: 0.98,
    value_action_samples=8,
    policy_learn_rate=lambda batch: 0.0001,
    policy_action_samples=32,

    profile=False,
    log_level=1
)
