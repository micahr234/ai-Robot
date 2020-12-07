import torch
import math
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from execute import Execute
import numpy as np

num_of_states = 6
state_frames = 3
num_of_actions = 2
action_distributions = 5

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states * state_frames, 512),
    torch.nn.ReLU(),
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

def scale(tensor, input_min, input_max):
    slope = (input_max - input_min) * 0.5
    intercept = (input_max + input_min) * 0.5
    scaled_tensor = (tensor - intercept) / slope
    return scaled_tensor

def state_input_transform(state):
    xform_state = torch.Tensor([state]).squeeze(2).contiguous()
    xform_state = scale(xform_state, torch.tensor([-1.0]*num_of_states), torch.tensor([1.0]*num_of_states))
    return xform_state

def reward_input_transform(reward):
    xform_reward = torch.Tensor([[reward]]).contiguous()
    return xform_reward

def action_input_transform(action):
    xform_action = torch.Tensor([action]).contiguous()
    return xform_action

def survive_input_transform(done):
    xform_survive = torch.Tensor([[float(done)]]).contiguous()
    return xform_survive

def action_output_transform(action):
    xform_action = action.tolist()
    return xform_action

unity_env = UnityEnvironment('./environments/CubeChase/CubeChase.exe', base_port=10000 + np.random.randint(1000), seed=1, no_graphics=False, side_channels=[])
gym_env = UnityToGymWrapper(unity_env, False, False, True)

Execute(
    instance_name='CubeChase1',

    environment_name='CubeChase',
    environment=gym_env,
    state_input_transform=state_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    episode_timestamp=False,
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
    action_distributions=action_distributions,

    value_learn_rate=lambda batch: 0.001,
    value_next_learn_factor=lambda batch: 0.8,
    value_action_samples=8,
    value_discount=0.99,
    policy_learn_rate=lambda batch: 0.0001,
    policy_action_samples=32,

    profile=False,
    log_level=1,
    gpu=None
)
