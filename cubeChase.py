import torch
import math
from execute import Execute

action_shape = [2]
state_shape = [9]
num_of_latent_states = 9
num_of_random_states = num_of_latent_states

preprocess_fwd_net = torch.nn.Sequential(
    torch.nn.Linear(state_shape[0], 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, num_of_latent_states * 2)
)
preprocess_rev_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, state_shape[0])
)
policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states + num_of_random_states, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, action_shape[0])
)
value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states + action_shape[0], 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

def state_input_transform(state):
    xform_state = torch.FloatTensor([state]).contiguous()
    return xform_state

def reward_input_transform(reward):
    xform_reward = torch.FloatTensor([[reward]]).contiguous()
    return xform_reward

def action_input_transform(action):
    xform_action = torch.FloatTensor(action).contiguous()
    return xform_action

def action_output_transform(action):
    xform_action = [action.tolist()]
    return xform_action

Execute(
    instance_name='CubeChase1',
    environment_name='CubeChase',
    agent_name='agent',
    profile=False,
    render=False,
    render_delay=0,
    verbosity=False,

    max_timestep=20000,
    learn_interval=200,
    save=False,
    episode_timestamp=True,
    batches=200,
    batch_size=200,
    memory_buffer_size=20000,

    preprocess_learn_rate=lambda batch: 0.001,
    preprocess_latent_learn_factor=lambda batch: 1.0,#0.9998**batch
    policy_value_learn_rate=lambda batch: 0.001,
    policy_entropy_learn_factor=lambda batch: 0.01 * (math.cos(2 * math.pi * (batch * 10) / 3000) + 1.0) / 2 * 0.9998 ** (batch * 10),
    policy_delay=10,
    value_learn_rate=lambda batch: 0.001,
    value_next_learn_factor=lambda batch: 0.8,
    discount=lambda batch: (1 - 0.9998**batch) * 0.1 + 0.9,

    action_shape=action_shape,
    state_shape=state_shape,
    num_of_latent_states=num_of_latent_states,
    num_of_random_states=num_of_random_states,

    preprocess_fwd_net=preprocess_fwd_net,
    preprocess_rev_net=preprocess_rev_net,
    policy_net=policy_net,
    value_net=value_net,

    state_input_transform=state_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    action_output_transform=action_output_transform
)
