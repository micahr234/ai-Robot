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

        tensor_board,

        action_random_prob,

        value_learn_rate,
        value_discount,
        value_polyak,

        policy_learn_rate,
        policy_polyak,

        gpu
        ):

        print('')
        print('Creating Model Free Deterministic Agent')

        self.name = str(name)
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.tensor_board = tensor_board
        self.action_random_prob = action_random_prob
        self.policy_learn_rate = policy_learn_rate
        self.value_learn_rate = value_learn_rate
        self.value_discount = value_discount
        self.policy_polyak = policy_polyak
        self.value_polyak = value_polyak

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'

        self.train_device = torch.device('cpu')
        if torch.cuda.is_available():
            if gpu is None:
                num_of_gpus = torch.cuda.device_count()
                gpu = int(torch.randint(low=0, high=num_of_gpus, size=[]))
            self.train_device = torch.device('cuda:' + str(gpu))
        self.act_device = torch.device('cpu')
        print('Train device:', self.train_device)
        print('Act device:', self.act_device)

        self.build_policy_network()
        self.build_value_network()

        self.policy_net_action_copy = copy.deepcopy(self.policy_net).to(self.act_device)

        self.learn_count = 1
        self.action_count = 1

    @staticmethod
    def sample_data(data, data_length, batch_size):

        index = torch.randint(0, data_length, (batch_size,))
        kwargs = {}
        for key, value in data.items():
            kwargs[key] = value[index, :]

        return kwargs

    @staticmethod
    def polyak_averaging(net, target_net, polyak):

        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

    def act(self, **observation):

        with torch.no_grad():

            self.policy_net_action_copy.eval()

            state = observation['observation_direct']
            state.to(self.act_device)

            action_temp = self.policy_net_action_copy(state)

            if torch.rand(()) > self.action_random_prob(self.action_count):
                action = action_temp
            else:
                action = torch.rand_like(action_temp) * 2 - 1

        self.action_count += 1

        return action

    def learn(self, data, batch_size, num_of_batches):

        for batch_num in range(1, num_of_batches + 1):

            # get data for batch
            with torch.no_grad():
                data_length = data['survive'].shape[0]
                batch_data = self.sample_data(data, data_length, batch_size)
                state = batch_data['observation_direct'].to(self.train_device, non_blocking=True)
                action = batch_data['action'].to(self.train_device, non_blocking=True)
                state_next = batch_data['observation_next_direct'].to(self.train_device, non_blocking=True)
                reward = batch_data['reward'].to(self.train_device, non_blocking=True)
                survive = batch_data['survive'].to(self.train_device, non_blocking=True)

            self.learn_value(state, action, state_next, reward, survive)
            self.learn_policy(state)

            self.learn_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())

    def learn_value(self, state_latent, action, state_next_latent, reward, survive):

        self.value_net.train()
        self.value_target_net.eval()
        self.policy_target_net.eval()

        # calculate intrinsic reward
        with torch.no_grad():
            reward_total = reward

        # value forward pass
        with torch.no_grad():
            action_next_hallu = self.policy_target_net(state_next_latent)

        value = self.value_net(state_latent, action)

        with torch.no_grad():
            value_next_target = self.value_target_net(state_next_latent, action_next_hallu)
            value_target = reward_total + value_next_target * survive * self.value_discount

        # optimize value
        self.value_optimizer.zero_grad()
        value_loss = torch.nn.functional.mse_loss(value, value_target)
        value_loss.backward()
        self.value_optimizer.step()

        # update target models
        self.polyak_averaging(self.value_net, self.value_target_net, self.value_polyak)

        # update value learn rates
        self.value_scheduler.step()

        # log value results
        with torch.no_grad():
            value_avg = torch.mean(value)
            value_target_avg = torch.mean(value_target)
        self.tensor_board.add_scalar('learn_value/avg', value_avg.item(), self.learn_count)
        self.tensor_board.add_scalar('learn_value/target_avg', value_target_avg.item(), self.learn_count)
        self.tensor_board.add_scalar('learn_value/loss', value_loss.item(), self.learn_count)
        #self.tensor_board.add_scalar('learn_value/learn_rate', self.value_scheduler.get_last_lr()[0], self.learn_count)

    def learn_policy(self, state_latent):

        self.value_net.eval()
        self.policy_net.train()

        # policy forward pass
        action = self.policy_net(state_latent)
        value = self.value_net(state_latent, action)

        # optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss = -torch.mean(value)
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target models
        self.polyak_averaging(self.policy_net, self.policy_target_net, self.policy_polyak)

        # update policy learn rates
        self.policy_scheduler.step()

        # log policy results
        self.tensor_board.add_scalar('learn_policy/loss', policy_loss.item(), self.learn_count)
        #self.tensor_board.add_scalar('learn_policy/learn_rate', self.policy_scheduler.get_last_lr()[0], self.learn_count)

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

        self.value_net = Value_Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_target_net = copy.deepcopy(self.value_net)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1.0)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, self.value_learn_rate, last_epoch=-1)

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

        self.policy_net = Policy_Net(self.policy_net_structure).to(self.train_device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_target_net = copy.deepcopy(self.policy_net)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1.0, weight_decay=0.0)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, self.policy_learn_rate, last_epoch=-1)
