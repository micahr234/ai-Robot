import torch
import math
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from execute import Execute
import numpy as np

num_of_states = 1
state_frames = 3
num_of_actions = 2
num_of_latent_states = 6

latent_net = torch.nn.Sequential(
    torch.nn.Conv2d(num_of_states, 32, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ELU(),
    torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ELU(),
    torch.nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ELU(),
    torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ELU(),
    torch.nn.Flatten(),
    torch.nn.Linear(2304, num_of_latent_states)
)

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames, 512),
    torch.nn.ReLU(),
    #torch.nn.Linear(512, 256),
    #torch.nn.ELU(),
    #torch.nn.Linear(256, 128),
    #torch.nn.Tanh(),
    torch.nn.Linear(512, num_of_actions)
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
    torch.nn.Linear(512, num_of_latent_states)
)

reward_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1)
)

survive_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1)
)

def scale(tensor, input_min, input_max):
    slope = (input_max - input_min) * 0.5
    intercept = (input_max + input_min) * 0.5
    scaled_tensor = (tensor - intercept) / slope
    return scaled_tensor

def state_input_transform(state):
    xform_state = torch.FloatTensor(state).permute(1, 0, 4, 2, 3).contiguous()
    xform_state = scale(xform_state, 0.0, 1.0)
    return xform_state

def reward_input_transform(reward):
    xform_reward = torch.FloatTensor([[reward]]).contiguous()
    return xform_reward

def action_input_transform(action):
    xform_action = torch.FloatTensor(action).contiguous()
    return xform_action

def survive_input_transform(done):
    xform_survive = torch.Tensor([[float(done)]]).contiguous()
    return xform_survive

def action_output_transform(action):
    xform_action = [action.tolist()]
    return xform_action

unity_env = UnityEnvironment('./environments/CubeChaseVisual/CubeChase.exe', base_port=10000 + np.random.randint(1000), seed=1, no_graphics=False, side_channels=[])
gym_env = UnityToGymWrapper(unity_env, True, False, False, True)

Execute(
    instance_name='CubeChaseVisual1',

    environment_name='CubeChaseVisual',
    environment=gym_env,
    state_input_transform=state_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    episode_timestamp=False,
    render=False,
    render_delay=0.0,

    agent_name='agent_model_based_deterministic_actor',
    max_timestep=20000,
    learn_interval=500,
    batches=500,
    batch_size=200,
    memory_buffer_size=20000,
    save=False,

    latent_net=latent_net,
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
    value_learn_rate=lambda batch: 0.001,
    value_next_learn_factor=lambda batch: 0.98,
    value_action_samples=8,
    value_action_samples_std=0.01,
    value_hallu_loops=1,
    policy_learn_rate=lambda batch: 0.0001,
    policy_action_samples=32,

    profile=False,
    log_level=1,
    gpu=0
)
