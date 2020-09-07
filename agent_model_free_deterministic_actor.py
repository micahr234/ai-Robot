from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
from memory import *
import numpy as np #for debug

# Define agent
class agent():

    # ------------------------- Initialization -------------------------

    def __init__(
                self,
                name,

                value_net,
                policy_net,

                tensor_board,
                state_input_transform,
                reward_input_transform,
                action_input_transform,
                survive_input_transform,
                action_output_transform,

                value_learn_rate,
                value_next_learn_factor,
                value_action_samples,
                value_action_samples_std,
                value_discount,
                policy_learn_rate,

                batches,
                batch_size,
                memory_buffer_size,
                log_level,
                gpu
                ):

        self.name = str(name)
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.tensor_board = tensor_board
        self.state_input_transform = state_input_transform
        self.reward_input_transform = reward_input_transform
        self.action_input_transform = action_input_transform
        self.survive_input_transform = survive_input_transform
        self.action_output_transform = action_output_transform
        self.batches = batches
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.log_level = log_level
        self.policy_learn_rate = policy_learn_rate
        self.value_learn_rate = value_learn_rate
        self.value_next_learn_factor = value_next_learn_factor
        self.value_action_samples = value_action_samples
        self.value_action_samples_std = value_action_samples_std
        self.value_discount = value_discount

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.latent_filename = self.memory_dir / 'latent.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.env_filename = self.memory_dir / 'env.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

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

        self.memory = Memory(self.memory_buffer_size, self.memory_buffer_filename)

        self.batch_count = 1
        self.record_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        with torch.no_grad():
            state = self.state_input_transform(in_state).detach().to(self.act_device)

            self.policy_net_action_copy.eval()

            action = self.policy_net_action_copy(state)

            out_action = self.action_output_transform(action[0, :].cpu())

        return out_action

    def record(self, in_state, in_action, in_reward, in_state_next, in_survive):

        with torch.no_grad():
            state = self.state_input_transform(in_state).cpu()
            action = self.action_input_transform(in_action).cpu()
            reward = self.reward_input_transform(in_reward).cpu()
            state_next = self.state_input_transform(in_state_next).cpu()
            survive = self.survive_input_transform(in_survive).cpu()
            self.memory.add(state, action, reward, state_next, survive)

        self.record_count += 1

        pass

    def learn(self):

        print('Agent ' + str(self.name) + ' learning fom ' + str(len(self.memory)) + ' samples')

        for batch_num in range(1, self.batches + 1):

            # get data for batch
            state, action, reward, state_next, survive = self.memory.sample(self.batch_size)
            state = state.to(self.train_device, non_blocking=True)
            action = action.to(self.train_device, non_blocking=True)
            reward = reward.to(self.train_device, non_blocking=True)
            state_next = state_next.to(self.train_device, non_blocking=True)
            survive = survive.to(self.train_device, non_blocking=True)

            self.learn_value(state, action, state_next, reward, survive)
            self.learn_policy(state)

            self.batch_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())

        # print summary
        print('Agent finished learning')

        pass

    def learn_value(self, state_latent, action, state_next_latent, reward, survive):

        self.value_net.train()
        self.policy_net.eval()

        # value forward pass
        value_buffer = []
        value_loss_buffer = []

        with torch.no_grad():
            state_latent_hallu = state_latent
            action_hallu = action
            state_next_latent_hallu, reward_hallu, survive_hallu = (state_next_latent, reward, survive)
            action_next_hallu = self.policy_net.sample(state_next_latent_hallu, samples=self.value_action_samples, std=self.value_action_samples_std)

        value_hallu = self.value_net(state_latent_hallu, action_hallu)
        value_next_hallu = self.value_net.forward_multi_action(state_next_latent_hallu, action_next_hallu)
        value_diff_hallu = value_hallu - value_next_hallu * survive_hallu * self.value_discount
        (reward_hallu, value_diff_hallu) = torch.broadcast_tensors(reward_hallu, value_diff_hallu)

        # optimize value
        self.value_optimizer.zero_grad()
        value_next_hallu.register_hook(lambda grad: grad * self.value_scheduler_next_learn_factor)
        value_loss = torch.nn.functional.mse_loss(value_diff_hallu, reward_hallu)
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            value_buffer.append(value_hallu)
            value_loss_buffer.append(value_loss.unsqueeze(0))

        # update value learn rates
        self.value_scheduler.step()
        self.value_scheduler_next_learn_factor = self.value_next_learn_factor(self.value_scheduler.last_epoch)

        # log value results
        if self.log_level >= 1:
            with torch.no_grad():
                value_avg = torch.mean(torch.cat(value_buffer, dim=0))
                value_loss_avg = torch.mean(torch.cat(value_loss_buffer, dim=0))
            self.tensor_board.add_scalar('learn_value/avg', value_avg.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_value/loss', value_loss_avg.item(), self.batch_count)

        # log value learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_value/learn_rate', self.value_scheduler.get_last_lr()[0], self.batch_count)
            self.tensor_board.add_scalar('learn_value/next_learn_factor', self.value_scheduler_next_learn_factor, self.batch_count)

        pass

    def learn_policy(self, state_latent):

        self.value_net.eval()
        self.policy_net.train()

        # policy forward pass
        action = self.policy_net(state_latent)
        value = self.value_net(state_latent, action)

        # optimize policy
        self.policy_optimizer.zero_grad()
        value_loss = -torch.mean(value)
        policy_loss = value_loss
        policy_loss.backward()

        self.policy_optimizer.step()

        # update policy learn rates
        self.policy_scheduler.step()

        # log policy results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_policy/value_loss', value_loss.item(), self.batch_count)

        # log policy learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_policy/learn_rate', self.policy_scheduler.get_last_lr()[0], self.batch_count)

        pass

    def save(self):

        print('Saving network and experience')
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_value_network(self):

        class Value_Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state, action):
                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                y = self.net(c)
                return y

            def forward_multi_action(self, state, multi_action):
                multi_count = multi_action.shape[0]
                multi_state = state.unsqueeze(0).repeat(multi_count, 1, 1, 1)

                multi_state_shape = list(multi_state.shape)
                del multi_state_shape[0]
                multi_state_shape[0] = -1
                multi_state = multi_state.view(multi_state_shape)

                multi_action_shape = list(multi_action.shape)
                del multi_action_shape[0]
                multi_action_shape[0] = -1
                multi_action = multi_action.view(multi_action_shape)

                multi_value = self.forward(multi_state, multi_action)

                multi_value_shape = list(multi_value.shape)
                multi_value_shape[0] = -1
                multi_value_shape.insert(0, multi_count)
                multi_value = multi_value.view(multi_value_shape)

                return multi_value

        self.value_net = Value_Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1.0)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, self.value_learn_rate, last_epoch=-1)
        self.value_scheduler_next_learn_factor = self.value_next_learn_factor(self.value_scheduler.last_epoch)

        pass

    def build_policy_network(self):

        class Policy_Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state):
                y = self.net(state.flatten(start_dim=1))

                if torch.isnan(y).sum() > 0:
                    raise ValueError('Nan values in actions')

                action = torch.tanh(y)
                return action

            def sample(self, state, samples=1, std=0.0):
                y = self.net(state.flatten(start_dim=1))

                if torch.isnan(y).sum() > 0:
                    raise ValueError('Nan values in actions')

                new_shape = [samples] + [1]*2
                y_repeated = y.unsqueeze(0).repeat(new_shape)
                y_repeated_noisy = y_repeated + torch.randn_like(y_repeated) * std
                action = torch.tanh(y_repeated_noisy)
                return action

        self.policy_net = Policy_Net(self.policy_net_structure).to(self.train_device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1.0, weight_decay=0.0)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, self.policy_learn_rate, last_epoch=-1)

        pass
