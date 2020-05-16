import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from memory import *

# Define agent
class Agent():

    # ------------------------- Initialization -------------------------

    def __init__(self,
                 name,
                 num_of_actions,
                 num_of_states,
                 num_of_latent_states,
                 preprocess_fwd_net,
                 preprocess_rev_net,
                 value_net,
                 policy_net,
                 tensor_board,
                 **params
                 ):

        self.name = str(name)
        self.num_of_actions = num_of_actions
        self.num_of_states = num_of_states
        self.num_of_latent_states = num_of_latent_states
        self.preprocess_fwd_net_structure = preprocess_fwd_net
        self.preprocess_rev_net_structure = preprocess_rev_net
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.tensor_board = tensor_board

        # Error if any of these variables do not exist in params
        assert 'verbosity' in params, '"verbosity" variable required'
        self.verbosity = params['verbosity']
        assert 'discount' in params, '"discount" variable required'
        self.discount = params['discount']
        assert 'value_learn_rate' in params, '"value_learn_rate" variable required'
        self.value_learn_rate = params['value_learn_rate']
        assert 'policy_learn_rate' in params, '"policy_learn_rate" variable required'
        self.policy_learn_rate = params['policy_learn_rate']
        assert 'preprocess_learn_rate' in params, '"preprocess_learn_rate" variable required'
        self.preprocess_learn_rate = params['preprocess_learn_rate']
        assert 'policy_delay' in params, '"policy_delay" variable required'
        self.policy_delay = params['policy_delay']
        assert 'preprocess_learn_beta' in params, '"preprocess_learn_beta" variable required'
        self.preprocess_learn_beta = params['preprocess_learn_beta']
        assert 'batches' in params, '"batches" variable required'
        self.batches = params['batches']
        assert 'memory_buffer_size' in params, '"memory_buffer_size" variable required'
        self.memory_buffer_size = params['memory_buffer_size']
        assert 'batch_size' in params, '"batch_size" variable required'
        self.batch_size = params['batch_size']
        assert 'next_learn_factor' in params, '"next_learn_factor" variable required'
        self.next_learn_factor = params['next_learn_factor']
        assert 'action_randomness' in params, '"action_randomness" variable required'
        self.action_randomness = params['action_randomness']

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.preprocess_filename = self.memory_dir / 'preprocess.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

        self.train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.act_device = torch.device('cpu')
        print('Train device:', self.train_device)
        print('Act device:', self.act_device)

        self.build_preprocess_network()
        self.build_value_network()
        self.build_policy_network()
        self.policy_net_action_copy = copy.deepcopy(self.policy_net).to(self.act_device)
        self.preprocess_net_action_copy = copy.deepcopy(self.preprocess_net).to(self.act_device)

        self.memory = Memory(self.memory_buffer_size, self.memory_buffer_filename, self.num_of_states, self.num_of_actions)

        self.batch_count = 1
        self.record_count = 1
        self.action_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        state = np.array(in_state, ndmin=2)
        state = torch.from_numpy(state).float().detach().to(self.act_device)

        self.preprocess_net.eval()
        self.policy_net.eval()

        with torch.no_grad():
            state_mu, state_logvar = self.preprocess_net_action_copy.encode(state)
            action = self.policy_net_action_copy(state_mu, a=self.action_randomness)[0]

        out_action = action.cpu().numpy()

        self.action_count += 1

        return out_action

    def record(self, in_state, in_action, in_reward, in_next_state, in_done):

        # Log experience
        if self.verbosity:
            for n in range(self.num_of_states):
                self.tensor_board.add_scalar('Record/state' + str(n), in_state[n], self.record_count)
                self.tensor_board.add_scalar('Record/next_state' + str(n), in_next_state[n], self.record_count)
            for n in range(self.num_of_actions):
                self.tensor_board.add_scalar('Record/action' + str(n), in_action[n], self.record_count)
            self.tensor_board.add_scalar('Record/reward', in_reward, self.record_count)
            self.tensor_board.add_scalar('Record/in_done', in_done, self.record_count)

        self.cumulative_reward += in_reward
        if in_done:
            self.tensor_board.add_scalar('Record/cumulative_reward', self.cumulative_reward, self.record_count)
            self.cumulative_reward = 0

        # Save memory
        state = np.array(in_state, ndmin=2)
        action = np.array(in_action, ndmin=2)
        reward = np.array(in_reward, ndmin=2)
        next_state = np.array(in_next_state, ndmin=2)
        done = np.array(in_done, ndmin=2)
        self.memory.add(state, action, reward, next_state, done)

        self.record_count += 1

        pass

    def learn(self):

        if len(self.memory) < self.batch_size:
            print('Agent waiting for more samples to learn from')
            return
        else:
            print('Agent ' + str(self.name) + ' learning fom ' + str(len(self.memory)) + ' samples')

        self.preprocess_net.train()
        self.value_net.train()
        self.policy_net.train()

        state_all_batches, action_all_batches, reward_all_batches, next_state_all_batches, done_all_batches = self.memory.sample(self.batch_size * self.batches)
        state_all_batches = state_all_batches.to(self.train_device, non_blocking=True)
        action_all_batches = action_all_batches.to(self.train_device, non_blocking=True)
        reward_all_batches = reward_all_batches.to(self.train_device, non_blocking=True)
        next_state_all_batches = next_state_all_batches.to(self.train_device, non_blocking=True)
        done_all_batches = done_all_batches.to(self.train_device, non_blocking=True)

        for batch_num in range(1, self.batches + 1):

            batch_LL = (batch_num - 1) * self.batches
            batch_UL = batch_num * self.batches
            state = state_all_batches[batch_LL:batch_UL, :]
            action = action_all_batches[batch_LL:batch_UL, :]
            reward = reward_all_batches[batch_LL:batch_UL, :]
            next_state = next_state_all_batches[batch_LL:batch_UL, :]
            done = done_all_batches[batch_LL:batch_UL, :]

            # preprocessor forward pass
            state_reconstructed, state_mu, state_logvar, state_latent = self.preprocess_net(state)
            with torch.no_grad():
                next_state_reconstructed, next_state_mu, next_state_logvar, next_state_latent = self.preprocess_net(next_state)

            def norm_mse_loss(input, target):
                target = target.detach()
                target_std, target_mean = torch.std_mean(target, dim=0)
                target_std = target_std + 1e-12
                target_normalized = (target - target_mean) / target_std
                loss = (target_normalized - input) ** 2
                loss_mean = torch.mean(loss)
                return loss_mean

            # optimize preprocessor
            self.preprocess_optimizer.zero_grad()
            preprocess_latent_loss = -0.5 * torch.mean(1 + state_logvar - state_mu.pow(2) - state_logvar.exp())
            preprocess_reconstruction_loss = norm_mse_loss(state_reconstructed, state)
            preprocess_loss = preprocess_reconstruction_loss + preprocess_latent_loss * self.preprocess_learn_beta
            preprocess_loss.backward()
            self.preprocess_optimizer.step()

            # log preprocessor results
            self.tensor_board.add_scalar('Learn_Preprocess/loss_latent', preprocess_latent_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Preprocess/loss_reconstruction', preprocess_reconstruction_loss.item(), self.batch_count)

            # value forward pass
            state_preprocessed, d = self.preprocess_net.encode(state)
            next_state_preprocessed, d = self.preprocess_net.encode(next_state)
            values = self.value_net(state_preprocessed, action)
            actions_next = self.policy_net(next_state_preprocessed, a=0.0)
            values_next = self.value_net(next_state_preprocessed, actions_next)
            values_diff = values - values_next * self.discount * (1.0 - done)
            value_avg = torch.mean(values).detach()

            # optimize value
            values_next.register_hook(lambda grad: grad * self.next_learn_factor)
            self.value_optimizer.zero_grad()
            value_loss = torch.nn.functional.mse_loss(values_diff, reward)
            value_loss.backward()
            self.value_optimizer.step()
            self.preprocess_optimizer2.step()

            # log value results
            self.tensor_board.add_scalar('Learn_Value/avg', value_avg.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Value/loss', value_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Value/loss_div_value_avg', value_loss.item() / value_avg.item(), self.batch_count)

            if self.batch_count % self.policy_delay == 0:
                # policy forward pass
                state_preprocessed, d = self.preprocess_net.encode(state)
                actions_current = self.policy_net(state_preprocessed.detach(), a=0.0)
                values = self.value_net(state_preprocessed, actions_current)
                #look into advantage

                # optimize policy
                self.policy_optimizer.zero_grad()
                policy_loss = -torch.mean(values)
                policy_loss.backward()
                self.policy_optimizer.step()

                # log policy results
                self.tensor_board.add_scalar('Learn_Policy/loss', policy_loss.item(), self.batch_count)

            self.batch_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())
        self.preprocess_net_action_copy.load_state_dict(self.preprocess_net.state_dict())

        # print summary
        print('Agent finished learning')

        pass

    def save(self):

        print('Saving network and experience')
        torch.save(self.preprocess_net.state_dict(), self.preprocess_filename)
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_preprocess_network(self):

        class Net(torch.nn.Module):

            def __init__(self, fwd_net, rev_net, num_of_latent_states):
                super().__init__()
                self.fwd_net = fwd_net
                self.rev_net = rev_net
                self.num_of_latent_states = num_of_latent_states
                pass

            def encode(self, state_input):
                x = self.fwd_net(state_input)
                mu = x[:, :self.num_of_latent_states]
                logvar = x[:, self.num_of_latent_states:]
                return mu, logvar

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                return z

            def decode(self, z):
                x = self.rev_net(z)
                return x

            def forward(self, state_input):
                mu, logvar = self.encode(state_input)
                z = self.reparameterize(mu, logvar)
                reconstructed_state_input = self.decode(z)
                return reconstructed_state_input, mu, logvar, z

        self.preprocess_net = Net(self.preprocess_fwd_net_structure, self.preprocess_rev_net_structure, self.num_of_latent_states).to(self.train_device)

        if self.preprocess_filename.is_file():
            # Load preprocess network
            print('Loading preprocess network from file ' + str(self.preprocess_filename))
            self.preprocess_net.load_state_dict(torch.load(self.preprocess_filename))

        else:
            # Build preprocess network
            print('No preprocess network loaded from file')

        self.preprocess_optimizer = torch.optim.Adam(self.preprocess_net.parameters(), lr=self.preprocess_learn_rate)
        self.preprocess_optimizer2 = torch.optim.Adam(self.preprocess_net.parameters(), lr=self.preprocess_learn_rate)

        pass

    def build_value_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                x = torch.cat((state_input, action_input), dim=-1)
                x = self.net(x)
                return x

        self.value_net = Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, a=0.0):
                x = self.net(state_input)
                x = x + a * torch.randn_like(x, requires_grad=False)
                x = torch.nn.functional.hardtanh(x)
                return x

        self.policy_net = Net(self.policy_net_structure).to(self.train_device)

        if self.policy_filename.is_file():
            # Load value network
            print('Loading max policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learn_rate)

        pass
