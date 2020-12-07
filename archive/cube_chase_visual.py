import torch
import math
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import execute

num_of_indirect_observations = 1
num_of_indirect_states = 4
num_of_direct_observations = 1
num_of_latent_states = num_of_indirect_states + num_of_direct_observations
state_frames = 3
num_of_actions = 2


class pixel_to_pos(torch.nn.Module):

    def __init__(self):

        super(pixel_to_pos, self).__init__()

        self.conv11 = torch.nn.Conv2d(num_of_indirect_observations, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv21 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv22 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.drop23 = torch.nn.Dropout2d(p=0.2)

        self.conv31 = torch.nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv32 = torch.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv33 = torch.nn.Conv2d(16, num_of_indirect_states, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv42 = torch.nn.Conv2d(num_of_indirect_states, num_of_indirect_states, kernel_size=(84, 84), stride=(1, 1), padding=(0, 0), groups=num_of_indirect_states)
        self.flat43 = torch.nn.Flatten()

        self.debug_img = None

    def forward(self, img):

        x = self.conv11(img)
        x = torch.nn.functional.elu(x)
        x = self.conv12(x)
        x = torch.nn.functional.elu(x)

        x = self.conv21(x)
        x = torch.nn.functional.elu(x)
        x = self.conv22(x)
        x = torch.nn.functional.elu(x)
        x = self.drop23(x)

        x = self.conv31(x)
        x = torch.nn.functional.elu(x)
        x = self.conv32(x)
        x = torch.nn.functional.elu(x)
        x = self.conv33(x)
        x = torch.nn.functional.elu(x)

        y = self.conv42(x)
        output = self.flat43(y)

        self.debug_img = x.clone()

        return output

latent_net = pixel_to_pos()

policy_net = torch.nn.Sequential(
    torch.nn.Linear(num_of_latent_states * state_frames, 512),
    torch.nn.ReLU(),
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

def observation_input_transform(state):
    state_direct_xform = torch.FloatTensor([])
    state_indirect_xform = torch.FloatTensor(state[0]).permute(2, 0, 1).contiguous()
    state_indirect_target_xform = torch.FloatTensor(state[1]).contiguous()
    #state_indirect_target_xform = torch.zeros_like(state_indirect_xform).repeat([2, 1, 1])
    #x = -int(torch.clamp(state_indirect_temp[1] * 6.0 + 42.0, min=0, max=83))
    #y = int(torch.clamp(state_indirect_temp[0] * 6.0 + 42.0, min=0, max=83))
    #state_indirect_target_xform[0, x, y] = 1.0
    #x = -int(torch.clamp(state_indirect_temp[3] * 6.0 + 42.0, min=0, max=83))
    #y = int(torch.clamp(state_indirect_temp[2] * 6.0 + 42.0, min=0, max=83))
    #state_indirect_target_xform[1, x, y] = 1.0
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

unity_env = UnityEnvironment('environments/CubeChaseVisual/CubeChase.exe', base_port=10000, worker_id=np.random.randint(1000), no_graphics=False)
gym_env = UnityToGymWrapper(unity_env, False, False, True)

execute.execute(
    instance_name='CubeChaseVisual1',

    environment_name='CubeChaseVisual',
    environment=gym_env,
    observation_input_transform=observation_input_transform,
    reward_input_transform=reward_input_transform,
    action_input_transform=action_input_transform,
    survive_input_transform=survive_input_transform,
    action_output_transform=action_output_transform,
    max_episode_timestamp=200,
    render_delay=0.0,

    agent_name='agent_image_test',
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
    action_random_prob=lambda i: 0.5 * (i < 5000),

    latent_learn_rate=lambda i: 0.0001,#0.0001
    model_learn_rate=lambda i: 0.0001,
    reward_learn_rate=lambda i: 0.0001,
    survive_learn_rate=lambda i: 0.0001,
    value_learn_rate=lambda i: 0.001,
    value_hallu_loops=1,
    policy_learn_rate=lambda i: 0.0001,

    profile=False,
    log_level=1,
    gpu=None
)

gym_env.close()
