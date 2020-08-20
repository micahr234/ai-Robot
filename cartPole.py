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
num_of_latent_states = num_of_states * 2

latent_fwd_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states, num_of_latent_states)
)

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions * 3 * 5)
)

value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

model_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_latent_states * 2)
)

reward_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_latent_states * 2 + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1 * 2)
)

survive_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_latent_states * 2 + num_of_actions, 512),
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
    xform_state = scale(xform_state, torch.tensor([-0.2]*4+[1.0]), torch.tensor([0.2]*4+[200.0]))
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

    agent_name='agent',
    max_timestep=20000,
    learn_interval=500,
    batches=500,
    batch_size=200,
    memory_buffer_size=20000,
    save=False,

    latent_fwd_net=latent_fwd_net,
    model_net=model_net,
    reward_net=reward_net,
    survive_net=survive_net,
    value_net=value_net,
    policy_net=policy_net,
    state_frames=state_frames,
    latent_states=num_of_latent_states,

    latent_learn_rate=lambda batch: 0.0001,
    model_learn_rate=lambda batch: 0.0001,
    reward_learn_rate=lambda batch: 0.0001,
    survive_learn_rate=lambda batch: 0.0001,
    value_learn_rate=lambda batch: 0.0003,
    value_next_learn_factor=lambda batch: 0.98,
    policy_learn_rate=lambda batch: 0.0001,
    policy_learn_entropy_factor=lambda batch: 0.0001,
    policy_action_samples=20,

    profile=False,
    log_level=1
)
