import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


# Define reward prediction network
class AgentTorchContinuous():

    # ------------------------- Initialization -------------------------

    def __init__(self, name, action_space_min, action_space_max, state_space_min, state_space_max, reward_space_min, reward_space_max,
                 discount=0.9999, batch_size=1000, value_learn_rate=0.0001, policy_learn_rate=0.0001, policy_copy_rate=0.0001,
                 learn_iterations=10, memory_buffer_size=0, next_learn_factor=0.1, debug=False):

        self.debug = debug

        self.name = name

        print('Creating agent ' + str(self.name))

        self.value_filename = name + '_value' + '.pt'
        self.policy_filename = name + '_policy' + '.pt'
        self.max_policy_filename = name + '_max_policy' + '.pt'
        self.memory_buffer_filename = name + '_data' + '.pt'

        self.discount = discount
        self.value_learn_rate = value_learn_rate
        self.policy_copy_rate = policy_copy_rate
        self.policy_learn_rate = policy_learn_rate
        self.learn_iterations = learn_iterations
        self.max_size_of_memory_buffer = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor

        # dim1 = min/max of each variables
        # example [10,2,100]
        self.action_space_min = action_space_min
        self.action_space_max = action_space_max
        self.num_of_actions = len(self.action_space_min)
        self.action_space_min_array = np.array(self.action_space_min)
        self.action_space_max_array = np.array(self.action_space_max)

        # dim1 = min/max of each variables
        # example [10,2,100]
        self.state_space_min = state_space_min
        self.state_space_max = state_space_max
        self.num_of_states = len(self.state_space_min)
        self.state_space_min_array = np.array(self.state_space_min)
        self.state_space_max_array = np.array(self.state_space_max)

        # dim1 = min/max of each variables
        # example [10,2,100]
        self.reward_space_min = reward_space_min
        self.reward_space_max = reward_space_max
        self.reward_space_min_array = np.array(self.reward_space_min)
        self.reward_space_max_array = np.array(self.reward_space_max)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)

        self.build_value_network()
        self.build_policy_network()

        if self.max_size_of_memory_buffer > 0:
            self.create_memory_buffer()
            self.load_memory_buffer()

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state, use_max_policy=False):

        in_state = np.array(in_state, ndmin=2)
        in_state = self.scale(in_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        state = torch.from_numpy(in_state).float().detach().to(self.device)

        self.policy.eval()
        self.max_policy.eval()

        with torch.no_grad():
            if use_max_policy:
                action = self.max_policy.forward(state)
            else:
                action = self.policy.forward(state)

        out_action = action.cpu().numpy()
        out_action = self.scale(out_action, -1, 1, self.action_space_min_array, self.action_space_max_array)
        #out_action = np.squeeze(out_action, axis=0)

        return out_action

    def record(self, in_state, in_action, in_reward, in_next_state, in_done, in_timestep):

        in_state = np.array(in_state, ndmin=2)
        in_state = self.scale(in_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_state = torch.from_numpy(in_state).float().detach().to(self.device)
        in_action = np.array(in_action, ndmin=2)
        in_action = self.scale(in_action, self.action_space_min_array, self.action_space_max_array, -1, 1)
        in_action = torch.from_numpy(in_action).float().detach().to(self.device)
        in_reward = np.array(in_reward, ndmin=2)
        in_reward = self.scale(in_reward, self.reward_space_min_array, self.reward_space_max_array, -1, 1)
        in_reward = torch.from_numpy(in_reward).float().detach().to(self.device)
        in_next_state = np.array(in_next_state, ndmin=2)
        in_next_state = self.scale(in_next_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_next_state = torch.from_numpy(in_next_state).float().detach().to(self.device)
        in_done = np.array(in_done, ndmin=2)
        in_done = torch.from_numpy(in_done).float().detach().to(self.device)
        self.save_memory(in_state, in_action, in_reward, in_next_state, in_done)

        pass

    def learn(self):

        state, action, reward, next_state, done = self.recall_memory()

        print('Agent ' + str(self.name) + ' learning fom ' + str(state.shape[0]) + ' samples')

        trainset = torch.utils.data.TensorDataset(state, action, next_state, done, reward)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=False)

        if self.debug: start_time = time.time(); elapsed_time = time.time() - start_time; print('Started at: ' + str(elapsed_time))

        for epoch in range(self.learn_iterations):

            if self.debug: elapsed_time = time.time() - start_time; print('Begin epoch: ' + str(elapsed_time))

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            running_count = 0.0

            for i, data in enumerate(trainloader, 0):

                if self.debug: elapsed_time = time.time() - start_time; print('Begin batch: ' + str(elapsed_time))

                # get the inputs; data is a list of [inputs, labels]
                in_state, in_action, in_next_state, in_done, out_reward = data

                # set the model to train mode
                self.value.train()
                self.max_policy.train()

                if self.debug: elapsed_time = time.time() - start_time; print('Begin forward: ' + str(elapsed_time))

                # forward pass
                values = self.value(in_state, in_action)

                max_policy_actions = self.max_policy.forward(in_next_state)
                values_next_with_grad = self.value(in_next_state, max_policy_actions)
                values_next = self.next_learn_factor * values_next_with_grad \
                              + (1.0 - self.next_learn_factor) * values_next_with_grad.detach()

                values_diff = values - values_next * self.discount * (1.0 - in_done)

                if self.debug: elapsed_time = time.time() - start_time; print('Begin opto value: ' + str(elapsed_time))

                # optimize value
                self.value_optimizer.zero_grad()
                value_loss = self.value_criterion(values_diff, out_reward)
                value_loss.backward(retain_graph=True)
                self.value_optimizer.step()

                if self.debug: elapsed_time = time.time() - start_time; print('Begin opto policy: ' + str(elapsed_time))

                # optimize max policy
                self.max_policy_optimizer.zero_grad()
                policy_loss = self.max_policy_criterion(values_next_with_grad)
                policy_loss.backward()
                self.max_policy_optimizer.step()

                if self.debug: elapsed_time = time.time() - start_time; print('Begin copy policy: ' + str(elapsed_time))

                # copy policy
                policy_dict = self.policy.state_dict()
                max_policy_dict = self.max_policy.state_dict()
                for param_name in self.policy.state_dict():
                    policy_dict[param_name] = (1 - self.policy_copy_rate) * policy_dict[
                        param_name] + self.policy_copy_rate * max_policy_dict[param_name]
                    pass
                self.policy.load_state_dict(policy_dict)

                if self.debug: elapsed_time = time.time() - start_time; print('Gather stats: ' + str(elapsed_time))

                # gather statistics
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                running_count += 1.0

                if self.debug: elapsed_time = time.time() - start_time; print('End batch: ' + str(elapsed_time))

            if self.debug: elapsed_time = time.time() - start_time; print('End epoch: ' + str(elapsed_time))

            print('Epoch: ' + str(epoch + 1)
                  + ' \t\tvalue loss:' + str(epoch_value_loss / running_count)
                  + ' \t\tpolicy loss:' + str(epoch_policy_loss / running_count))

        self.save_networks()
        self.save_memory_buffer()

        pass

    # ------------------------- Network Functions -------------------------

    def build_value_network(self):

        print('Building value network')

        class Net(nn.Module):

            def __init__(self, num_of_states, num_of_actions):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(num_of_states + num_of_actions, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, 1)

            def forward(self, state_input, action_input):
                x = torch.cat((state_input, action_input), 1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.value = Net(self.num_of_states, self.num_of_actions).to(self.device)

        value_file = Path(self.value_filename)

        if value_file.is_file():
            # Load value network
            print('Loading value network from file ' + self.value_filename)
            self.value.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_criterion = nn.MSELoss()
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        print('Building policy networks')

        class Net(nn.Module):

            def __init__(self, num_of_states, num_of_actions):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(num_of_states, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, num_of_actions)

            def forward(self, state_input):
                x = F.relu(self.fc1(state_input))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.policy = Net(self.num_of_states, self.num_of_actions).to(self.device)

        policy_file = Path(self.policy_filename)

        if policy_file.is_file():
            # Load value network
            print('Loading policy network from file ' + self.policy_filename)
            self.policy.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No policy network loaded from file')

        self.max_policy = Net(self.num_of_states, self.num_of_actions).to(self.device)

        max_policy_file = Path(self.max_policy_filename)

        if max_policy_file.is_file():
            # Load value network
            print('Loading max policy network from file ' + self.max_policy_filename)
            self.max_policy.load_state_dict(torch.load(self.max_policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        def maximize_loss(output):
            loss = -torch.mean(output)
            return loss

        self.max_policy_criterion = maximize_loss
        self.max_policy_optimizer = optim.Adam(self.max_policy.parameters(), lr=self.policy_learn_rate)

        pass

    def save_networks(self):

        torch.save(self.value.state_dict(), self.value_filename)
        torch.save(self.policy.state_dict(), self.policy_filename)
        torch.save(self.max_policy.state_dict(), self.max_policy_filename)

        pass

    # ------------------------- Memory Buffer -------------------------

    def create_memory_buffer(self):

        #self.experience_mem_index = 0
        #self.memory_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_states])
        #self.memory_action = np.zeros([self.max_size_of_memory_buffer, self.num_of_actions])
        #self.memory_reward = np.zeros([self.max_size_of_memory_buffer, 1])
        #self.memory_next_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_states])
        #self.memory_done = np.zeros([self.max_size_of_memory_buffer, 1])

        self.experience_mem_index = 0
        self.memory_state = torch.empty([self.max_size_of_memory_buffer, self.num_of_states]).to(self.device)
        self.memory_action = torch.empty([self.max_size_of_memory_buffer, self.num_of_actions]).to(self.device)
        self.memory_reward = torch.empty([self.max_size_of_memory_buffer, 1]).to(self.device)
        self.memory_next_state = torch.empty([self.max_size_of_memory_buffer, self.num_of_states]).to(self.device)
        self.memory_done = torch.empty([self.max_size_of_memory_buffer, 1]).to(self.device)

    def load_memory_buffer(self):

        experience_buffer_file = Path(self.memory_buffer_filename)

        if experience_buffer_file.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + self.memory_buffer_filename)
            experience_buffer = torch.load(self.memory_buffer_filename)

            temp = experience_buffer['state']
            temp_index = temp.shape[0]
            if temp_index > self.max_size_of_memory_buffer:
                raise ValueError('Experience memory buffer overflow.')

            self.experience_mem_index = temp_index
            self.memory_state[0:temp_index, :] = experience_buffer['state']
            self.memory_action[0:temp_index, :] = experience_buffer['action']
            self.memory_reward[0:temp_index, :] = experience_buffer['reward']
            self.memory_next_state[0:temp_index, :] = experience_buffer['next_state']
            self.memory_done[0:temp_index, :] = experience_buffer['done']

        else:

            print('No experience buffer to load')

        pass

    def save_memory_buffer(self):

        state, action, reward, next_state, done = self.recall_memory()

        experience_buffer_file = Path(self.memory_buffer_filename)
        torch.save({'state':state, 'action':action, 'reward':reward, 'next_state':next_state, 'done':done}, experience_buffer_file)

        pass

    def save_memory(self, state, action, reward, next_state, done):

        if self.experience_mem_index >= self.max_size_of_memory_buffer:
            self.memory_state[:-1, :] = self.memory_state[1:, :]
            self.memory_action[:-1, :] = self.memory_action[1:, :]
            self.memory_reward[:-1, :] = self.memory_reward[1:, :]
            self.memory_next_state[:-1, :] = self.memory_next_state[1:, :]
            self.memory_done[:-1, :] = self.memory_done[1:, :]
            self.experience_mem_index -= 1

        index = self.experience_mem_index
        self.memory_state[index, :] = state
        self.memory_action[index, :] = action
        self.memory_reward[index, :] = reward
        self.memory_next_state[index, :] = next_state
        self.memory_done[index, :] = done
        self.experience_mem_index += 1

        pass

    def recall_memory(self):

        index = self.experience_mem_index
        state = self.memory_state[0:index, :]
        action = self.memory_action[0:index, :]
        reward = self.memory_reward[0:index, :]
        next_state = self.memory_next_state[0:index, :]
        done = self.memory_done[0:index, :]

        return state, action, reward, next_state, done

    # ------------------------- Normalization -------------------------

    def scale(self, input, input_min, input_max, output_min, output_max):

        input_scaled = (input - input_min) / (input_max - input_min)
        output = input_scaled * (output_max - output_min) + output_min

        return output