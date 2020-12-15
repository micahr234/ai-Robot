from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
from memory import *
import numpy as np  # for debug


# Define agent
class model_free_deterministic_agent():

    def __init__(
        self,
        name,
        value_net,
        policy_net,
        value_learn_rate,
        value_discount,
        value_polyak,
        policy_learn_rate,
        policy_polyak):

        print('')
        print('Creating Model Free Deterministic Agent')

        self.name = str(name)
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.policy_learn_rate = policy_learn_rate
        self.value_learn_rate = value_learn_rate
        self.value_discount = value_discount
        self.policy_polyak = policy_polyak
        self.value_polyak = value_polyak

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'

        self.device = torch.device('cpu')

        self.build_policy_network()
        self.build_value_network()

    @staticmethod
    def polyak_averaging(net, target_net, polyak):

        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

    def set_device(self, device):

        with torch.no_grad():

            self.device = device

            self.value_net.to(self.device)
            self.value_target_net.to(self.device)
            self.policy_net.to(self.device)
            self.policy_target_net.to(self.device)

    def act(self, action_random_prob, **data):

        with torch.no_grad():

            self.policy_net.eval()

            state = data['observation_direct']

            action_temp = self.policy_net(state)

            if torch.rand(()) > action_random_prob:
                action = action_temp
            else:
                action = torch.rand_like(action_temp) * 2 - 1

        return action

    def learn(self, **data):

        state = data['observation_direct']
        action = data['action']
        state_next = data['observation_next_direct']
        reward = data['reward']
        survive = data['survive']

        (value_avg, value_loss) = self.learn_value(state, action, state_next, reward, survive)
        (policy_loss) = self.learn_policy(state)

        return value_avg, value_loss, policy_loss

    def learn_value(self, state, action, state_next, reward, survive):

        self.value_net.train()
        self.value_target_net.eval()
        self.policy_target_net.eval()

        # calculate intrinsic reward
        with torch.no_grad():
            reward_total = reward

        # value forward pass
        with torch.no_grad():
            action_next = self.policy_target_net(state_next)

        value = self.value_net(state, action)

        with torch.no_grad():
            value_next_target = self.value_target_net(state_next, action_next)
            value_target = reward_total + value_next_target * survive * self.value_discount

        # optimize value
        self.value_optimizer.zero_grad()
        value_loss = torch.nn.functional.mse_loss(value, value_target)
        value_loss.backward()
        self.value_optimizer.step()

        # update target models
        self.polyak_averaging(self.value_net, self.value_target_net, self.value_polyak)

        # calculate the average value
        with torch.no_grad():
            value_avg = torch.mean(value)

        return value_avg, value_loss

    def learn_policy(self, state):

        self.value_net.eval()
        self.policy_net.train()

        # policy forward pass
        action = self.policy_net(state)
        value = self.value_net(state, action)

        # optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(value)
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target models
        self.polyak_averaging(self.policy_net, self.policy_target_net, self.policy_polyak)

        return policy_loss

    def save(self):

        print('Saving network and experience')
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)

    def build_value_network(self):

        class Value_Net(torch.nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, state, action):

                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                y = self.net(c)

                return y

        self.value_net = Value_Net(self.value_net_structure).to(self.device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_target_net = copy.deepcopy(self.value_net)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.policy_learn_rate)

    def build_policy_network(self):

        class Policy_Net(torch.nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, state):

                y = self.net(state.flatten(start_dim=1))

                if torch.isnan(y).sum() > 0:
                    raise ValueError('Nan values in actions')

                action = torch.tanh(y)

                return action

        self.policy_net = Policy_Net(self.policy_net_structure).to(self.device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_target_net = copy.deepcopy(self.policy_net)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.value_learn_rate)
