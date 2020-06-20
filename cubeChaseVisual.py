import torch
import math
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from execute import Execute

num_of_states = 1
previous_states = 3
num_of_actions = 2
num_of_latent_states = 20
num_of_random_states = num_of_latent_states

class Flatten(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        pass
    def forward(self, input):
        return input.view([input.size(0)] + self.size).contiguous()

class Unflatten(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        pass
    def forward(self, input):
        return input.view([input.size(0)] + self.size).contiguous()

preprocess_fwd_net = torch.nn.Sequential(
    torch.nn.Conv2d(num_of_states, 32, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ReLU(),
    Flatten([2304]),
    torch.nn.Linear(2304, num_of_latent_states * 2),
)
preprocess_rev_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states, 2304),
    torch.nn.ReLU(),
    Unflatten([256, 3, 3]),
    torch.nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2)),
    torch.nn.ReLU(),
    torch.nn.ConvTranspose2d(32, num_of_states, kernel_size=(4, 4), stride=(2, 2))
)
policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * previous_states + num_of_random_states, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions)
)
value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * previous_states + num_of_actions, 512),
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
    xform_state = torch.FloatTensor(state).permute(1, 0, 4, 2, 3).contiguous()
    xform_state = scale(xform_state, 0.0, 1.0)
    return xform_state

def reward_input_transform(reward):
    xform_reward = torch.FloatTensor([[reward]]).contiguous()
    return xform_reward

def action_input_transform(action):
    xform_action = torch.FloatTensor(action).contiguous()
    return xform_action

def done_input_transform(done):
    xform_reward = torch.Tensor([[float(done)]]).contiguous()
    return xform_reward

def action_output_transform(action):
    xform_action = [action.tolist()]
    return xform_action

unity_env = UnityEnvironment('./environments/CubeChaseVisual/CubeChase.exe', base_port=2001, seed=1, no_graphics=False, side_channels=[])
gym_env = UnityToGymWrapper(unity_env, True, False, False, True)

Execute(
    instance_name='CubeChaseVisual1',
    environment_name='CubeChaseVisual',
    environment=gym_env,
    previous_states=previous_states,
    agent_name='agent',
    profile=False,
    render=False,
    render_delay=0,
    verbosity=False,

    max_timestep=100000,
    learn_interval=200,
    save=True,
    episode_timestamp=False,
    batches=200,
    batch_size=2000,
    memory_buffer_size=20000,

    preprocess_learn_rate=lambda batch: 0.0001,
    preprocess_latent_learn_factor=lambda batch: 1.0,
    policy_value_learn_rate=lambda batch: 0.0001,
    policy_entropy_learn_factor=lambda batch: 0.01, # 0.01 * (math.cos(2 * math.pi * (batch * 10) / 3000) + 1.0) / 2 * 0.9998 ** (batch * 10),
    value_learn_rate=lambda batch: 0.0001,
    value_next_learn_factor=lambda batch: 0.8,
    discount=lambda batch: 0.99,  # (1 - 0.9998**batch) * 0.1 + 0.9,

    num_of_latent_states=num_of_latent_states,
    num_of_random_states=num_of_random_states,

    preprocess_fwd_net=preprocess_fwd_net,
    preprocess_rev_net=preprocess_rev_net,
    policy_net=policy_net,
    value_net=value_net,

    state_input_transform=state_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    done_input_transform=done_input_transform,
    action_output_transform=action_output_transform
)
