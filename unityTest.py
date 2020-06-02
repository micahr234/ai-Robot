import torch
from execute import Execute

num_of_actions = 2
num_of_states = 8 + 1
num_of_latent_states = 9

preprocess_fwd_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_states, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, num_of_latent_states * 2)
)
preprocess_rev_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, num_of_states)
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
policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions * 2)
)

Execute(
    instance_name='Unity1',
    environment_name='Unity',
    agent_name='agent',
    profile=False,
    render=False,
    render_delay=0,
    verbosity=False,

    max_timestep=30000,
    learn_interval=200,
    save=False,
    episode_timestamp=True,

    batches=200,
    batch_size=2000,
    memory_buffer_size=100000,
    value_learn_rate=0.001,
    policy_learn_rate=0.001,
    preprocess_learn_rate=0.001,
    discount=0.99,
    policy_delay=10,
    preprocess_learn_beta=1.0,
    next_learn_factor=0.8,
    policy_learn_beta=0.00001,
    value_learn_rate_schedule=lambda epoch: 0.99**(epoch),
    policy_learn_rate_schedule=lambda epoch: 0.99**(epoch),
    preprocess_learn_rate_schedule=lambda epoch: (abs(-(epoch+1) % 30) / 30) * 0.99**(epoch),

    num_of_actions=num_of_actions,
    num_of_states=num_of_states,
    num_of_latent_states=num_of_latent_states,

    preprocess_fwd_net=preprocess_fwd_net,
    preprocess_rev_net=preprocess_rev_net,
    value_net=value_net,
    policy_net=policy_net
)
