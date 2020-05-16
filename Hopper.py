import torch
from execute import Execute

num_of_actions = 3
num_of_states = 15 + 1
num_of_latent_states = 30

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
    torch.nn.Linear(128, num_of_actions),
    torch.nn.Tanh()
)

Execute(
    instance_name='Hopper1',
    environment_name='HopperBulletEnv-v0',
    agent_name='agent',
    profile=False,
    render=False,
    render_delay=0,
    verbosity=False,

    max_timestep=10000000,
    learn_interval=2000,
    save=False,
    episode_timestamp=True,
    action_noise=0.1,

    batches=200,
    batch_size=4000,
    memory_buffer_size=500000,
    value_learn_rate=0.001,
    policy_learn_rate=0.001,
    preprocess_learn_rate=0.001,
    discount=1.0,
    policy_delay=10,
    preprocess_learn_beta=2.0,
    next_learn_factor=0.9,
    action_randomness=0.1,

    num_of_actions=num_of_actions,
    num_of_states=num_of_states,
    num_of_latent_states=num_of_latent_states,

    preprocess_fwd_net=preprocess_fwd_net,
    preprocess_rev_net=preprocess_rev_net,
    value_net=value_net,
    policy_net=policy_net
)
