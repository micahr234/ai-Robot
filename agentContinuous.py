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

    def __init__(self, name, action_type, num_of_action_values, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                 batch_size=1000, learn_iterations=10, memory_buffer_size=100000,
                 discount=0.999, value_learn_rate=0.0001, policy_learn_rate=0.00001, next_learn_factor=0.0,
                 debug=False):

        self.debug = debug

        self.name = str(name)

        print('Creating agent ' + self.name)

        self.memory_dir = Path.cwd() / 'memory' / name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

        self.discount = discount
        self.value_learn_rate = value_learn_rate
        self.policy_learn_rate = policy_learn_rate
        self.learn_iterations = learn_iterations
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor
        self.unquantize_actions = True if action_type == 'discrete' else False

        # dim1 = variables
        # example [5, 3, 3]
        self.action_space_min = action_space_min
        self.action_space_max = action_space_max
        self.num_of_actions = len(self.action_space_max)
        self.action_space_min_array = np.array(self.action_space_min)
        self.action_space_max_array = np.array(self.action_space_max)

        # dim1 = min/max of each variables
        # example [10,2,100]
        self.state_space_min = state_space_min
        self.state_space_max = state_space_max
        self.num_of_states = len(self.state_space_max)
        self.state_space_min_array = np.array(self.state_space_min)
        self.state_space_max_array = np.array(self.state_space_max)

        # dim1 = min/max of each variables
        # example [100]
        self.reward_space_min = reward_space_min
        self.reward_space_max = reward_space_max
        self.reward_space_min_array = np.array(self.reward_space_min)
        self.reward_space_max_array = np.array(self.reward_space_max)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.build_value_network()
        self.build_policy_network()

        self.memory = ExperienceMemory(self.memory_buffer_size, self.memory_buffer_filename, self.num_of_states, self.num_of_actions)

        self.tensor_board_dir = Path.cwd() / 'runs' / name / str(time.time())
        self.tensor_board = SummaryWriter(self.tensor_board_dir, max_queue=10000, flush_secs=120)
        hyper_params = {'agent_type': 'continuous',
                        'unquantize_actions': self.unquantize_actions,
                        'batch_size': self.batch_size,
                        'learn_iterations': self.learn_iterations,
                        'memory_buffer_size': self.memory_buffer_size,
                        'discount': self.discount,
                        'value_learn_rate': self.value_learn_rate,
                        'policy_learn_rate': self.policy_learn_rate,
                        'next_learn_factor': self.next_learn_factor}
        self.tensor_board.add_text('Hyper Params', str(hyper_params), 0)

        self.batch_count = 1
        self.record_count = 1
        self.action_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        state = np.array(in_state, ndmin=2)
        state = self.scale(state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        state = torch.from_numpy(state).float().detach().to(self.device)

        self.policy.eval()

        with torch.no_grad():
            action = self.policy(state)[0]

        out_action = action.cpu().numpy()
        out_action = self.scale(out_action, -1, 1, self.action_space_min_array, self.action_space_max_array)

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
        state = self.scale(state, self.state_space_min_array, self.state_space_max_array, -1, 1)

        action = np.array(in_action, ndmin=1)
        action = self.scale(action, self.action_space_min_array, self.action_space_max_array, -1, 1)
        action = np.array(action, ndmin=2)

        reward = np.array(in_reward, ndmin=2)
        reward = self.scale(reward, self.reward_space_min_array, self.reward_space_max_array, -1, 1)

        next_state = np.array(in_next_state, ndmin=2)
        next_state = self.scale(next_state, self.state_space_min_array, self.state_space_max_array, -1, 1)

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

        value_loss_results = []
        policy_loss_results = []

        for batch_num in range(1, self.learn_iterations + 1):

            state, action, reward, next_state, done = self.memory.sample(self.batch_size)

            # set the model to train mode
            self.value.train()
            self.policy.train()

            # value forward pass
            values = self.value(state, action)
            next_actions = self.policy(next_state)
            values_next = self.value(next_state, next_actions)
            values_diff = values - values_next * self.discount * (1.0 - done)

            # optimize value
            values_next.register_hook(lambda grad: grad * self.next_learn_factor)
            self.value_optimizer.zero_grad()
            value_loss = self.value_criterion(values_diff, reward)
            value_loss.backward()
            self.value_optimizer.step()

            # log value results
            value_loss_results.append(value_loss.item())
            self.tensor_board.add_scalar('Learn/value_loss', value_loss.item(), self.batch_count)

            # policy forward pass
            current_actions = self.policy(state)
            values = self.value(state, current_actions)

            # optimize policy
            self.max_policy_optimizer.zero_grad()
            policy_loss = self.max_policy_criterion(values)
            policy_loss.backward()
            self.max_policy_optimizer.step()

            # log policy results
            policy_loss_results.append(policy_loss.item())
            self.tensor_board.add_scalar('Learn/policy_loss', policy_loss.item(), self.batch_count)

            self.batch_count += 1

        # print summary
        print('Batches: ' + str(self.learn_iterations)
              + ' \t\tValue loss mean:' + str(np.mean(value_loss_results))
              + ' \t\tValue loss std:' + str(np.std(value_loss_results))
              + ' \t\tPolicy loss mean:' + str(np.mean(policy_loss_results))
              + ' \t\tPolicy loss std:' + str(np.std(policy_loss_results)))

        pass

    def save(self):

        print('Saving network and experience')
        self.save_networks()
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_value_network(self):

        print('Building value network')

        class Net(torch.nn.Module):

            def __init__(self, num_of_states, num_of_actions):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(num_of_states + num_of_actions, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc3 = torch.nn.Linear(128, 64)
                self.fc4 = torch.nn.Linear(64, 32)
                self.fc5 = torch.nn.Linear(32, 1)

            def forward(self, state_input, action_input):
                x = torch.cat((state_input, action_input), 1)
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))
                x = torch.nn.functional.relu(self.fc3(x))
                x = torch.nn.functional.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.value = Net(self.num_of_states, self.num_of_actions).to(self.device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_criterion = torch.nn.MSELoss()
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        print('Building policy network')

        class Net(torch.nn.Module):

            def __init__(self, num_of_states, num_of_actions):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(num_of_states, 256)
                self.fc2 = torch.nn.Linear(256, 128)
                self.fc3 = torch.nn.Linear(128, 64)
                self.fc4 = torch.nn.Linear(64, 32)
                self.fc5 = torch.nn.Linear(32, num_of_actions)

            def forward(self, state_input):
                x = torch.nn.functional.relu(self.fc1(state_input))
                x = torch.nn.functional.relu(self.fc2(x))
                x = torch.nn.functional.relu(self.fc3(x))
                x = torch.nn.functional.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.policy = Net(self.num_of_states, self.num_of_actions).to(self.device)

        if self.policy_filename.is_file():
            # Load value network
            print('Loading max policy network from file ' + str(self.policy_filename))
            self.policy.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        def maximize_loss(output):
            loss = -torch.mean(output)
            return loss

        self.max_policy_criterion = maximize_loss
        self.max_policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_learn_rate)

        pass

    def save_networks(self):

        torch.save(self.value.state_dict(), self.value_filename)
        torch.save(self.policy.state_dict(), self.policy_filename)

        pass


    # ------------------------- Normalization -------------------------

    def scale(self, input, input_min, input_max, output_min, output_max):

        input_scaled = (input - input_min) / (input_max - input_min)
        output = input_scaled * (output_max - output_min) + output_min

        return output
