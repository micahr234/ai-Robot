import torch
import math
from execute import Execute

num_of_actions = 3
num_of_states = 15 + 1
num_of_latent_states = 30
num_of_random_states = 4

preprocess_fwd_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, num_of_latent_states * 2)
)
preprocess_rev_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, num_of_states)
)
policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states + num_of_random_states, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions)
)
value_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states + num_of_actions, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
)

Execute(
    instance_name='Hopper1',
    environment_name='HopperBulletEnv-v0',
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
    memory_buffer_size=100000,

    preprocess_learn_rate=lambda batch: 0.001,
    preprocess_latent_learn_factor=lambda batch: 1.0,  # 0.9998**batch
    policy_value_learn_rate=lambda batch: 0.001,
    policy_entropy_learn_factor=lambda batch: 0.01 * (
                math.cos(2 * math.pi * (batch * 10) / 3000) + 1.0) / 2 * 0.9998 ** (batch * 10),
    policy_delay=10,
    value_learn_rate=lambda batch: 0.001,
    value_next_learn_factor=lambda batch: 0.8,
    discount=lambda batch: (1 - 0.9998 ** batch) * 0.1 + 0.9,

    num_of_actions=num_of_actions,
    num_of_states=num_of_states,
    num_of_latent_states=num_of_latent_states,
    num_of_random_states=num_of_random_states,

    preprocess_fwd_net=preprocess_fwd_net,
    preprocess_rev_net=preprocess_rev_net,
    policy_net=policy_net,
    value_net=value_net
)
