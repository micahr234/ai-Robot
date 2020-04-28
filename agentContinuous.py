import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from experienceMemory import *

#Add tanh to output to limit output

# Define agent
class AgentContinuous():

    # ------------------------- Initialization -------------------------

    def __init__(self, name, num_of_actions, num_of_states,
                 value_hidden_layer_sizes=[256, 128, 64, 32], policy_hidden_layer_sizes=[256, 128, 64, 32], preprocess_hidden_layer_sizes=[20, 50],
                 batch_size=1000, learn_iterations=10, memory_buffer_size=100000,
                 discount=1.0, value_learn_rate=0.0001, policy_learn_rate=0.0001, preprocess_learn_rate=0.0001, policy_delay=10, next_learn_factor=0.8, randomness=0.1,
                 debug=False):

        self.debug = debug

        self.name = str(name)

        print('Creating agent ' + self.name)

        self.memory_dir = Path.cwd() / 'memory' / name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.preprocess_filename = self.memory_dir / 'preprocess.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

        self.discount = discount
        self.value_learn_rate = value_learn_rate
        self.policy_learn_rate = policy_learn_rate
        self.preprocess_learn_rate = preprocess_learn_rate  # needs work
        self.policy_delay = policy_delay
        self.learn_iterations = learn_iterations
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor
        self.randomness = randomness

        self.num_of_actions = num_of_actions
        self.num_of_states = num_of_states

        self.preprocess_layer_sizes = [self.num_of_states] + preprocess_hidden_layer_sizes
        self.value_layer_sizes = [self.preprocess_layer_sizes[-1] + self.num_of_actions] + value_hidden_layer_sizes + [1]
        self.policy_layer_sizes = [self.preprocess_layer_sizes[-1]] + policy_hidden_layer_sizes + [self.num_of_actions]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.build_preprocess_network()
        self.build_value_network()
        self.build_policy_network()

        self.memory = ExperienceMemory(self.memory_buffer_size, self.memory_buffer_filename, self.num_of_states, self.num_of_actions)

        self.tensor_board_dir = Path.cwd() / 'runs' / name / str(time.time())
        self.tensor_board = SummaryWriter(self.tensor_board_dir, max_queue=10000, flush_secs=120)
        hyper_params = {'agent_type': 'continuous',
                        'batch_size': self.batch_size,
                        'learn_iterations': self.learn_iterations,
                        'memory_buffer_size': self.memory_buffer_size,
                        'discount': self.discount,
                        'value_learn_rate': self.value_learn_rate,
                        'policy_learn_rate': self.policy_learn_rate,
                        'preprocess_learn_rate': self.preprocess_learn_rate,
                        'policy_delay': self.policy_delay,
                        'next_learn_factor': self.next_learn_factor,
                        'randomness': self.randomness,
                        'preprocess_layer_sizes': self.preprocess_layer_sizes,
                        'value_layer_sizes': self.value_layer_sizes,
                        'policy_layer_sizes': self.policy_layer_sizes}
        self.tensor_board.add_text('Hyper Params', str(hyper_params), 0)

        self.batch_count = 1
        self.record_count = 1
        self.action_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        state = np.array(in_state, ndmin=2)
        state = torch.from_numpy(state).float().detach().to(self.device)

        self.preprocess.eval()
        self.policy.eval()

        with torch.no_grad():
            state_reconstructed_norm, state_mu, state_logvar, state_latent = self.preprocess(state)
            action = self.policy(state_mu, a=self.randomness)[0]

        out_action = action.cpu().numpy()

        self.action_count += 1

        return out_action

    def record(self, in_state, in_action, in_reward, in_next_state, in_done):

        # Log experience
        if self.debug:
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

        # set the model to train mode
        self.preprocess.train()
        self.value.train()
        self.policy.train()

        for batch_num in range(1, self.learn_iterations + 1):

            state, action, reward, next_state, done = self.memory.sample(self.batch_size)

            # preprocessor forward pass
            state_reconstructed, state_mu, state_logvar, state_latent = self.preprocess(state)
            state_preprocessed = state_mu.detach()
            with torch.no_grad():
                next_state_reconstructed, next_state_mu, next_state_logvar, next_state_latent = self.preprocess(next_state)
                next_state_preprocessed = next_state_mu.detach()

            #to be tested
            def c_loss(input, target):
                td = target.detach()
                target_std = torch.std(td, dim=0)
                #print(target_std)
                input.register_hook(lambda grad: print(grad))
                input.register_hook(lambda grad: grad / (target_std ** 2))
                input.register_hook(lambda grad: print(grad))
                loss = (td - input) ** 2
                loss_mean = torch.mean(loss)
                return loss_mean

            # optimize preprocessor
            self.preprocess_optimizer.zero_grad()
            preprocess_latent_loss = -0.5 * torch.mean(1 + state_logvar - state_mu.pow(2) - state_logvar.exp())
            preprocess_reconstruction_loss = c_loss(state_reconstructed, state)
            preprocess_loss = preprocess_reconstruction_loss #+ preprocess_latent_loss
            preprocess_loss.backward()
            self.preprocess_optimizer.step()

            # log preprocessor results
            self.tensor_board.add_scalar('Learn_Preprocess/loss_latent', preprocess_latent_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Preprocess/loss_reconstruction', preprocess_reconstruction_loss.item(), self.batch_count)

            # value forward pass
            values = self.value(state_preprocessed, action)
            actions_next = self.policy(next_state_preprocessed, a=0.0)
            values_next = self.value(next_state_preprocessed, actions_next)
            values_diff = values - values_next * self.discount * (1.0 - done)
            value_avg = torch.mean(values).detach()

            # optimize value
            values_next.register_hook(lambda grad: grad * self.next_learn_factor)
            self.value_optimizer.zero_grad()
            value_loss = torch.nn.functional.mse_loss(values_diff, reward)
            value_loss.backward()
            self.value_optimizer.step()

            # log value results
            self.tensor_board.add_scalar('Learn_Value/avg', value_avg.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Value/loss', value_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('Learn_Value/loss_div_value_avg', value_loss.item() / value_avg.item(), self.batch_count)

            if self.batch_count % self.policy_delay == 0:
                # policy forward pass
                actions_current = self.policy(state_preprocessed, a=0.0)
                values = self.value(state_preprocessed, actions_current)
                #look into advantage

                # optimize policy
                self.policy_optimizer.zero_grad()
                policy_loss = -torch.mean(values)
                policy_loss.backward()
                self.policy_optimizer.step()

                # log policy results
                self.tensor_board.add_scalar('Learn_Policy/loss', policy_loss.item(), self.batch_count)

            self.batch_count += 1

        # print summary
        print('Agent finished learning')

        pass

    def save(self):

        print('Saving network and experience')
        torch.save(self.preprocess.state_dict(), self.preprocess_filename)
        torch.save(self.value.state_dict(), self.value_filename)
        torch.save(self.policy.state_dict(), self.policy_filename)
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_preprocess_network(self):

        class Net(torch.nn.Module):

            def __init__(self, layer_sizes):
                super(Net, self).__init__()

                linear_layers = [torch.nn.Linear(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])]
                all_layers = []
                for layer in linear_layers[:-1]:
                    all_layers.append(layer)
                    all_layers.append(torch.nn.ReLU())
                self.fwd_layers = torch.nn.Sequential(*all_layers)

                self.fwd_layers_mu = torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
                )

                self.fwd_layers_logvar = torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
                )

                linear_layers = [torch.nn.Linear(in_f, out_f) for in_f, out_f in zip(layer_sizes[1:], layer_sizes[:-1])]
                linear_layers_reversed = list(reversed(linear_layers))
                all_layers = []
                for layer in linear_layers_reversed[:-1]:
                    all_layers.append(layer)
                    all_layers.append(torch.nn.ReLU())
                all_layers.append(linear_layers_reversed[-1])
                self.rev_layers = torch.nn.Sequential(*all_layers)

                pass

            def encode(self, state_input):
                x = self.fwd_layers(state_input)
                mu = self.fwd_layers_mu(x)
                logvar = self.fwd_layers_logvar(x)
                return mu, logvar

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                return z

            def decode(self, z):
                x = self.rev_layers(z)
                return x

            def forward(self, state_input):
                mu, logvar = self.encode(state_input)
                z = self.reparameterize(mu, logvar)
                reconstructed_state_input = self.decode(z)
                return reconstructed_state_input, mu, logvar, z

        self.preprocess = Net(self.preprocess_layer_sizes).to(self.device)

        if self.preprocess_filename.is_file():
            # Load preprocess network
            print('Loading preprocess network from file ' + str(self.preprocess_filename))
            self.preprocess.load_state_dict(torch.load(self.preprocess_filename))

        else:
            # Build preprocess network
            print('No preprocess network loaded from file')

        self.preprocess_optimizer = torch.optim.Adam(self.preprocess.parameters(), lr=self.preprocess_learn_rate)

        pass

    def build_value_network(self):

        class Net(torch.nn.Module):

            def __init__(self, layer_sizes):
                super(Net, self).__init__()
                linear_layers = [torch.nn.Linear(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])]
                all_layers = []
                for layer in linear_layers[:-1]:
                    all_layers.append(layer)
                    all_layers.append(torch.nn.ReLU())
                all_layers.append(linear_layers[-1])
                self.layers = torch.nn.Sequential(*all_layers)
                pass

            def forward(self, state_input, action_input):
                x = torch.cat((state_input, action_input), dim=-1)
                x = self.layers(x)
                return x

        self.value = Net(self.value_layer_sizes).to(self.device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        class Net(torch.nn.Module):

            def __init__(self, layer_sizes):
                super(Net, self).__init__()
                linear_layers = [torch.nn.Linear(in_f, out_f) for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:])]
                all_layers = []
                for layer in linear_layers[:-1]:
                    all_layers.append(layer)
                    all_layers.append(torch.nn.ReLU())
                all_layers.append(linear_layers[-1])
                all_layers.append(torch.nn.Tanh())
                self.layers = torch.nn.Sequential(*all_layers)
                pass

            def forward(self, state_input, a=0.0):
                x = self.layers(state_input)
                x = x + a * torch.randn_like(x, requires_grad=False)
                x = torch.nn.functional.hardtanh(x)
                return x

        self.policy = Net(self.policy_layer_sizes).to(self.device)

        if self.policy_filename.is_file():
            # Load value network
            print('Loading max policy network from file ' + str(self.policy_filename))
            self.policy.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_learn_rate)

        pass
