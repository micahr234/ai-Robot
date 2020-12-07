import torch
import math
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import execute

num_of_indirect_observations = 0
num_of_indirect_states = 0
num_of_direct_observations = 4
num_of_latent_states = num_of_indirect_states + num_of_direct_observations
state_frames = 3
num_of_actions = 2
action_distributions = 7

latent_net = torch.nn.Sequential(
)

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_of_actions * action_distributions)
)

policy_mix_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, action_distributions)
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

def observation_input_transform(state):
    state_direct_xform = torch.FloatTensor(state[0]).contiguous()
    state_indirect_xform = torch.FloatTensor([])
    state_indirect_target_xform = torch.FloatTensor([])
    return state_direct_xform, state_indirect_xform, state_indirect_target_xform

def reward_input_transform(reward):
    reward_xform = torch.FloatTensor([reward]).contiguous()
    return reward_xform

def action_input_transform(action):
    action_xform = torch.FloatTensor(action).contiguous()
    return action_xform

def survive_input_transform(done):
    survive_xform = torch.FloatTensor([float(done)]).contiguous()
    return survive_xform

def action_output_transform(action):
    action_xform = [action.tolist()]
    return action_xform

unity_env = UnityEnvironment('environments/CubeChase/CubeChase.exe', base_port=10000, worker_id=np.random.randint(1000), no_graphics=False)
gym_env = UnityToGymWrapper(unity_env, False, False, True)

execute.execute(
    instance_name='CubeChase1',

    environment_name='CubeChase',
    environment=gym_env,
    observation_input_transform=observation_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    max_episode_timestamp=None,
    render_delay=0.0,

    agent_name='agent_model_based_stochastic_actor',
    max_timestep=40000,
    learn_interval=500,
    batches=500,
    batch_size=200,
    memory_buffer_size=40000,
    save=False,

    latent_net=latent_net,
    model_net=model_net,
    reward_net=reward_net,
    survive_net=survive_net,
    value_net=value_net,
    policy_net=policy_net,
    policy_mix_net=policy_mix_net,
    state_frames=state_frames,
    latent_states=num_of_latent_states,
    action_distributions=action_distributions,

    latent_learn_rate=lambda i: 0.0001,
    latent_polyak=0.0025,
    model_learn_rate=lambda i: 0.0001,
    reward_learn_rate=lambda i: 0.0001,
    survive_learn_rate=lambda i: 0.0001,
    env_polyak=0.0025,
    value_learn_rate=lambda i: 0.0001,
    value_hallu_loops=1,
    value_polyak=0.0025,
    policy_learn_rate=lambda i: 0.0001,
    policy_polyak=0.00025,

    profile=False,
    log_level=1,
    gpu=None
)

gym_env.close()