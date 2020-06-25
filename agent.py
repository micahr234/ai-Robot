import numpy
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from memory import *

# Define agent
class Agent():

    # ------------------------- Initialization -------------------------

    def __init__(
                self,
                name,
                latent_states,

                latent_fwd_net,
                latent_rev_net,
                model_net,
                reward_net,
                terminate_net,
                value_net,
                policy_net,

                tensor_board,
                state_input_transform,
                reward_input_transform,
                action_input_transform,
                terminate_input_transform,
                action_output_transform,

                latent_learn_rate,
                latent_latent_learn_factor,

                model_learn_rate,
                reward_learn_rate,
                terminate_learn_rate,
                value_learn_rate,
                value_next_learn_factor,
                discount,
                policy_learn_rate,

                batches,
                batch_size,
                memory_buffer_size,
                verbosity,
                ):

        self.name = str(name)
        self.latent_states = latent_states
        self.latent_fwd_net_structure = latent_fwd_net
        self.latent_rev_net_structure = latent_rev_net
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.model_net_structure = model_net
        self.reward_net_structure = reward_net
        self.terminate_net_structure = terminate_net
        self.tensor_board = tensor_board
        self.state_input_transform = state_input_transform
        self.reward_input_transform = reward_input_transform
        self.action_input_transform = action_input_transform
        self.terminate_input_transform = terminate_input_transform
        self.action_output_transform = action_output_transform

        # Error if any of these variables do not exist in params
        self.batches = batches
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.verbosity = verbosity
        self.latent_learn_rate = latent_learn_rate
        self.latent_latent_learn_factor = latent_latent_learn_factor
        self.policy_learn_rate = policy_learn_rate
        self.value_learn_rate = value_learn_rate
        self.value_next_learn_factor = value_next_learn_factor
        self.discount = discount
        self.model_learn_rate = model_learn_rate
        self.reward_learn_rate = reward_learn_rate
        self.terminate_learn_rate = terminate_learn_rate

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.latent_filename = self.memory_dir / 'latent.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.model_filename = self.memory_dir / 'model.pt'
        self.reward_filename = self.memory_dir / 'reward.pt'
        self.terminate_filename = self.memory_dir / 'terminate.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

        self.train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.act_device = torch.device('cpu')
        print('Train device:', self.train_device)
        print('Act device:', self.act_device)

        self.build_latent_network()
        self.build_model_network()
        self.build_reward_network()
        self.build_terminate_network()
        self.build_policy_network()
        self.build_value_network()
        self.latent_net_action_copy = copy.deepcopy(self.latent_net).to(self.act_device)
        self.policy_net_action_copy = copy.deepcopy(self.policy_net).to(self.act_device)

        self.memory = Memory(self.memory_buffer_size, self.memory_buffer_filename)

        self.batch_count = 1
        self.record_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        state = self.state_input_transform(in_state).detach().to(self.act_device)

        self.latent_net_action_copy.eval()
        self.policy_net_action_copy.eval()

        with torch.no_grad():
            state_mu = self.latent_net_action_copy(state).flatten(start_dim=1, end_dim=2)
            action = self.policy_net_action_copy(state_mu)

        out_action = self.action_output_transform(action[0].cpu())

        return out_action

    def record(self, in_state, in_action, in_reward, in_state_next, in_terminate):

        state = self.state_input_transform(in_state).detach().cpu()
        action = self.action_input_transform(in_action).detach().cpu()
        reward = self.reward_input_transform(in_reward).detach().cpu()
        state_next = self.state_input_transform(in_state_next).detach().cpu()
        terminate = self.terminate_input_transform(in_terminate).detach().cpu()
        self.memory.add(state, action, reward, state_next, terminate)

        self.record_count += 1

        pass

    def learn(self):

        if len(self.memory) < self.batch_size:
            print('Agent waiting for more samples to learn from')
            return
        else:
            print('Agent ' + str(self.name) + ' learning fom ' + str(len(self.memory)) + ' samples')

        self.latent_net.train()
        self.value_net.train()
        self.policy_net.train()

        for batch_num in range(1, self.batches + 1):

            # get data for batch
            state, action, reward, state_next, terminate = self.memory.sample(self.batch_size)
            state = state.to(self.train_device, non_blocking=True)
            action = action.to(self.train_device, non_blocking=True)
            reward = reward.to(self.train_device, non_blocking=True)
            state_next = state_next.to(self.train_device, non_blocking=True)
            terminate = terminate.to(self.train_device, non_blocking=True)

            self.learn_latent(state, state_next)
            self.learn_model(state, action, state_next)
            self.learn_reward(state, action, reward)
            self.learn_terminate(state, action, terminate)
            self.learn_value(state, action)
            self.learn_policy(state)

            self.batch_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())
        self.latent_net_action_copy.load_state_dict(self.latent_net.state_dict())

        # print summary
        print('Agent finished learning')

        pass

    def learn_latent(self, state, state_next):

        # get latent learn rates
        self.latent_scheduler.step()
        latent_learn_rate = self.latent_scheduler.get_last_lr()[0]
        latent_latent_learn_factor = self.latent_latent_learn_factor(self.latent_scheduler.last_epoch)
        self.tensor_board.add_scalar('learn_latent/learn_rate', latent_learn_rate, self.batch_count)
        self.tensor_board.add_scalar('learn_latent/latent_learn_factor', latent_latent_learn_factor, self.batch_count)

        # learn latent
        if latent_learn_rate > 0.0:

            # latentor forward pass
            if self.batch_count % 2 == 0:
                state_last_frame = state[:, -1, :]
            else:
                state_last_frame = state_next[:, -1, :]
            state_reconstructed, state_mu, state_logvar = self.latent_net.loop(state_last_frame)

            # optimize latent
            self.latent_optimizer.zero_grad()
            latent_reconstruction_loss = torch.nn.functional.mse_loss(state_reconstructed, state_last_frame)
            latent_latent_loss = -0.5 * torch.mean(1 + state_logvar - state_mu.pow(2) - state_logvar.exp())
            latent_latent_loss.register_hook(lambda grad: grad * latent_latent_learn_factor)
            latent_loss = latent_reconstruction_loss + latent_latent_loss
            latent_loss.backward()
            self.latent_optimizer.step()

            # log latentor results
            self.tensor_board.add_scalar('learn_latent/reconstruction_loss', latent_reconstruction_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_latent/latent_loss', latent_latent_loss.item(), self.batch_count)

    def learn_model(self, state, action, state_next):

        # get model learn rates
        self.model_scheduler.step()
        model_learn_rate = self.model_scheduler.get_last_lr()[0]
        self.tensor_board.add_scalar('learn_model/learn_rate', model_learn_rate, self.batch_count)

        # learn model
        if model_learn_rate > 0.0:
            # model forward pass
            state_latent = self.latent_net(state).flatten(start_dim=1, end_dim=2)
            state_next_last_frame = state_next[:, [-1], :]
            state_next_latent_last_frame = self.latent_net(state_next_last_frame).flatten(start_dim=1, end_dim=2).detach()
            state_next_latent_last_frame_prediction = self.model_net(state_latent, action)

            # optimize model
            self.model_optimizer.zero_grad()
            model_loss = torch.nn.functional.mse_loss(state_next_latent_last_frame_prediction, state_next_latent_last_frame)
            model_loss.backward()
            self.model_optimizer.step()

            # log model results
            self.tensor_board.add_scalar('learn_model/loss', model_loss.item(), self.batch_count)

    def learn_reward(self, state, action, reward):
            
        # get reward learn rates
        self.reward_scheduler.step()
        reward_learn_rate = self.reward_scheduler.get_last_lr()[0]
        self.tensor_board.add_scalar('learn_reward/learn_rate', reward_learn_rate, self.batch_count)
    
        # learn reward
        if reward_learn_rate > 0.0:
            # reward forward pass
            state_latent = self.latent_net(state).flatten(start_dim=1, end_dim=2)
            reward_prediction = self.reward_net(state_latent, action)
    
            # optimize reward
            self.reward_optimizer.zero_grad()
            reward_loss = torch.nn.functional.mse_loss(reward_prediction, reward)
            reward_loss.backward()
            self.reward_optimizer.step()
    
            # log reward results
            self.tensor_board.add_scalar('learn_reward/loss', reward_loss.item(), self.batch_count)

    def learn_terminate(self, state, action, terminate):

        # get terminate learn rates
        self.terminate_scheduler.step()
        terminate_learn_rate = self.terminate_scheduler.get_last_lr()[0]
        self.tensor_board.add_scalar('learn_terminate/learn_rate', terminate_learn_rate, self.batch_count)

        # learn terminate
        if terminate_learn_rate > 0.0:
            # terminate forward pass
            state_latent = self.latent_net(state).flatten(start_dim=1, end_dim=2)
            terminate_prediction = self.terminate_net(state_latent, action)

            # optimize terminate
            self.terminate_optimizer.zero_grad()
            terminate_loss = torch.nn.functional.mse_loss(terminate_prediction, terminate)
            terminate_loss.backward()
            self.terminate_optimizer.step()

            # log terminate results
            self.tensor_board.add_scalar('learn_terminate/loss', terminate_loss.item(), self.batch_count)

    def learn_value(self, state, action):

        # get value learn rates
        self.value_scheduler.step()
        value_learn_rate = self.value_scheduler.get_last_lr()[0]
        value_next_learn_factor = self.value_next_learn_factor(self.value_scheduler.last_epoch)
        discount = self.discount(self.value_scheduler.last_epoch)
        self.tensor_board.add_scalar('learn_value/learn_rate', value_learn_rate, self.batch_count)
        self.tensor_board.add_scalar('learn_value/next_learn_factor', value_next_learn_factor, self.batch_count)
        self.tensor_board.add_scalar('learn_value/discount', discount, self.batch_count)

        # learn value
        if value_learn_rate > 0.0:

            # value forward pass
            state_latent = self.latent_net(state).flatten(start_dim=1, end_dim=2).detach()
            #state_next_latent = self.latent_net(state_next).flatten(start_dim=1, end_dim=2).detach()
            state_next_latent_last_frame_prediction = self.model_net(state_latent, action)
            state_next_latent_prediction = torch.cat((state[:, 1:, :], state_next_latent_last_frame_prediction.unsqueeze(1)), dim=1).flatten(start_dim=1, end_dim=2).detach()
            value = self.value_net(state_latent, action)
            action_next = self.policy_net(state_next_latent_prediction)
            value_next = self.value_net(state_next_latent_prediction, action_next)
            terminate_prediction = self.terminate_net(state_latent, action).detach()
            value_diff = value - value_next * discount * (1.0 - terminate_prediction)
            reward_prediction = self.reward_net(state_latent, action).detach()
            value_avg = torch.mean(value).detach()

            # optimize value
            self.value_optimizer.zero_grad()
            value_next.register_hook(lambda grad: grad * value_next_learn_factor)
            value_loss = torch.nn.functional.mse_loss(value_diff, reward_prediction)
            value_loss.backward()
            self.value_optimizer.step()

            # log value results
            self.tensor_board.add_scalar('learn_value/avg', value_avg.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_value/loss', value_loss.item(), self.batch_count)

    def learn_policy(self, state):
        
        # get policy learn rates
        self.policy_scheduler.step()
        policy_learn_rate = self.policy_scheduler.get_last_lr()[0]
        self.tensor_board.add_scalar('learn_policy/learn_rate', policy_learn_rate, self.batch_count)

        # learn policy
        if policy_learn_rate > 0.0:

            # policy forward pass
            with torch.no_grad():
                state_latent = self.latent_net(state).flatten(start_dim=1, end_dim=2)
            action_current = self.policy_net(state_latent)
            value = self.value_net(state_latent, action_current)

            # optimize policy
            self.policy_optimizer.zero_grad()
            policy_loss = -torch.mean(value)
            policy_loss.backward()
            self.policy_optimizer.step()

            # log policy results
            self.tensor_board.add_scalar('learn_policy/loss', policy_loss.item(), self.batch_count)

    def save(self):

        print('Saving network and experience')
        torch.save(self.latent_net.state_dict(), self.latent_filename)
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)
        torch.save(self.model_net.state_dict(), self.model_filename)
        torch.save(self.reward_net.state_dict(), self.reward_filename)
        torch.save(self.terminate_net.state_dict(), self.terminate_filename)
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_latent_network(self):

        class Net(torch.nn.Module):

            def __init__(self, fwd_net, rev_net, latent_states):
                super().__init__()
                self.fwd_net = fwd_net
                self.rev_net = rev_net
                self.latent_states = latent_states
                pass

            def encode(self, state_input):
                x = self.fwd_net(state_input)
                mu = x[:, :self.latent_states]
                logvar = x[:, self.latent_states:]
                return mu, logvar

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                return z

            def decode(self, z):
                r = self.rev_net(z)
                return r

            def loop(self, state_input):
                mu, logvar = self.encode(state_input)
                z = self.reparameterize(mu, logvar)
                r = self.decode(z)
                return r, mu, logvar

            def forward(self, multi_state_input):
                multi_mu = torch.empty(list(multi_state_input.shape[0:2]) + [self.latent_states], device=multi_state_input.device)
                for i in range(multi_state_input.shape[1]):
                    multi_mu[:, i, :], _ = self.encode(multi_state_input[:, i, :])
                return multi_mu

        self.latent_net = Net(self.latent_fwd_net_structure, self.latent_rev_net_structure, self.latent_states).to(self.train_device)

        if self.latent_filename.is_file():
            # Load latent network
            print('Loading latent network from file ' + str(self.latent_filename))
            self.latent_net.load_state_dict(torch.load(self.latent_filename))

        else:
            # Build latent network
            print('No latent network loaded from file')

        self.latent_optimizer = torch.optim.Adam(self.latent_net.parameters(), lr=1.0)
        self.latent_scheduler = torch.optim.lr_scheduler.LambdaLR(self.latent_optimizer, self.latent_learn_rate, last_epoch=-1)

        pass

    def build_policy_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input):
                x = self.net(state_input)
                return x

        self.policy_net = Net(self.policy_net_structure).to(self.train_device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1.0)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, self.policy_learn_rate, last_epoch=-1)

        pass

    def build_value_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                c = torch.cat((state_input, action_input), dim=-1)
                x = self.net(c)
                return x

        self.value_net = Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1.0)
        self.value_optimizer.add_param_group({'params': self.latent_net.fwd_net.parameters()})
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, self.value_learn_rate, last_epoch=-1)

        pass

    def build_model_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                c = torch.cat((state_input, action_input), dim=-1)
                x = self.net(c)
                return x

        self.model_net = Net(self.model_net_structure).to(self.train_device)

        if self.model_filename.is_file():
            # Load model network
            print('Loading model network from file ' + str(self.model_filename))
            self.model_net.load_state_dict(torch.load(self.model_filename))

        else:
            # Build model network
            print('No model network loaded from file')

        self.model_optimizer = torch.optim.Adam(self.model_net.parameters(), lr=1.0)
        self.model_scheduler = torch.optim.lr_scheduler.LambdaLR(self.model_optimizer, self.model_learn_rate, last_epoch=-1)

        pass

    def build_reward_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                c = torch.cat((state_input, action_input), dim=-1)
                x = self.net(c)
                return x

        self.reward_net = Net(self.reward_net_structure).to(self.train_device)

        if self.reward_filename.is_file():
            # Load reward network
            print('Loading reward network from file ' + str(self.reward_filename))
            self.reward_net.load_state_dict(torch.load(self.reward_filename))

        else:
            # Build reward network
            print('No reward network loaded from file')

        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=1.0)
        self.reward_scheduler = torch.optim.lr_scheduler.LambdaLR(self.reward_optimizer, self.reward_learn_rate, last_epoch=-1)

        pass

    def build_terminate_network(self):

        class Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                c = torch.cat((state_input, action_input), dim=-1)
                x = self.net(c)
                return x

        self.terminate_net = Net(self.terminate_net_structure).to(self.train_device)

        if self.terminate_filename.is_file():
            # Load terminate network
            print('Loading terminate network from file ' + str(self.terminate_filename))
            self.terminate_net.load_state_dict(torch.load(self.terminate_filename))

        else:
            # Build terminate network
            print('No terminate network loaded from file')

        self.terminate_optimizer = torch.optim.Adam(self.terminate_net.parameters(), lr=1.0)
        self.terminate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.terminate_optimizer, self.terminate_learn_rate, last_epoch=-1)

        pass