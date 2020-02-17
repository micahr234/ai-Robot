import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define reward prediction network
class AgentTorchDiscrete():

    # ------------------------- Initialization -------------------------

    def __init__(self, name, action_space, state_space_min, state_space_max, reward_space_min, reward_space_max,
                 discount=0.9999, batch_size=1000, value_learn_rate=0.0001, policy_learn_rate=0.0001,
                 policy_copy_rate=0.0001,
                 learn_iterations=10, memory_buffer_size=0, next_learn_factor=0.1):

        self.name = name

        print('Creating agent ' + str(self.name))

        self.value_filename = name + '_value' + '.pt'
        self.policy_filename = name + '_policy' + '.pt'
        self.max_policy_filename = name + '_max_policy' + '.pt'
        self.memory_buffer_filename = name + '.npz'

        self.discount = discount
        self.value_learn_rate = value_learn_rate
        self.policy_copy_rate = policy_copy_rate
        self.policy_learn_rate = policy_learn_rate
        self.learn_iterations = learn_iterations
        self.max_size_of_memory_buffer = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor

        # dim1 = variables, dim2 = list of possible actions for each variable
        # example [[0,1,2,3,4], [0,1,2], [0,1,2]]
        self.action_space = action_space
        self.num_of_actions = len(self.action_space)
        self.num_of_action_values = [len(i) for i in self.action_space]

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
        self.build_max_policy_network()

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
                policy_logits = self.max_policy.forward(state)
            else:
                policy_logits = self.policy.forward(state)

            policy_probs = F.softmax(policy_logits, dim=1)
            action = torch.multinomial(policy_probs, 1, replacement=True)[0]

        out_action = action.cpu().numpy()

        return out_action

    def record(self, in_state, in_action, in_reward, in_next_state, in_done):

        in_state = np.array(in_state, ndmin=2)
        in_state = self.scale(in_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_action = np.array(in_action, ndmin=2)
        in_reward = np.array(in_reward, ndmin=2)
        in_reward = self.scale(in_reward, self.reward_space_min_array, self.reward_space_max_array, -1, 1)
        in_next_state = np.array(in_next_state, ndmin=2)
        in_next_state = self.scale(in_next_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_done = np.array(in_done, ndmin=2)
        self.save_memory(in_state, in_action, in_reward, in_next_state, in_done)

        pass

    def learn(self):

        print('Agent ' + str(self.name) + ' learning')

        state, action, reward, next_state, done, validation = self.recall_memory()

        state = torch.from_numpy(state).float().detach().to(self.device)
        action = torch.from_numpy(action).float().detach().to(self.device)
        reward = torch.from_numpy(reward).float().detach().to(self.device)
        next_state = torch.from_numpy(next_state).float().detach().to(self.device)
        done = torch.from_numpy(done).float().detach().to(self.device)

        trainset = torch.utils.data.TensorDataset(state, action, next_state, done, reward)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.learn_iterations):

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            running_count = 0.0

            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                in_state, in_action, in_next_state, in_done, out_reward = data

                # set the model to train mode
                self.value.train()
                self.max_policy.train()

                # forward pass
                values = self.value(in_state)
                values_sum = torch.gather(values, 1, in_action.long())

                max_policy_logits = self.max_policy.forward(in_next_state)
                max_policy_probs = F.softmax(max_policy_logits.detach(), dim=1)
                values_next_with_grad = self.value(in_next_state)
                values_next = self.next_learn_factor * values_next_with_grad \
                              + (1.0 - self.next_learn_factor) * values_next_with_grad.detach()
                values_next_sum = torch.sum(values_next * max_policy_probs, 1, keepdim=True)

                values_diff = values_sum - values_next_sum * self.discount * (1.0 - in_done)

                # optimize value
                self.value_optimizer.zero_grad()
                value_loss = self.value_criterion(values_diff, out_reward)
                value_loss.backward(retain_graph=True)
                self.value_optimizer.step()

                policy_ground_truth = torch.argmax(values_next_with_grad.detach(), dim=1)

                # optimize max policy
                self.max_policy_optimizer.zero_grad()
                policy_loss = self.max_policy_criterion(max_policy_logits, policy_ground_truth)
                policy_loss.backward()
                self.max_policy_optimizer.step()

                # optimize policy
                policy_dict = self.policy.state_dict()
                max_policy_dict = self.max_policy.state_dict()
                for param_name in self.policy.state_dict():
                    policy_dict[param_name] = (1 - self.policy_copy_rate) * policy_dict[
                        param_name] + self.policy_copy_rate * max_policy_dict[param_name]
                    pass
                self.policy.load_state_dict(policy_dict)

                # gather statistics
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                running_count += 1.0

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

        self.value = Net(self.num_of_states, self.num_of_action_values[0]).to(self.device)

        value_file = Path(self.value_filename)

        if value_file.is_file():
            # Load value network
            print('Loading value network from file ' + self.value_filename)
            self.value.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_criterion = nn.MSELoss()
        # self.value_optimizer = optim.SGD(self.value.parameters(), lr=self.value_learn_rate, momentum=0)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        print('Building policy network')

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

        self.policy = Net(self.num_of_states, self.num_of_action_values[0]).to(self.device)

        policy_file = Path(self.policy_filename)

        if policy_file.is_file():
            # Load value network
            print('Loading policy network from file ' + self.policy_filename)
            self.policy.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No policy network loaded from file')

        pass

    def build_max_policy_network(self):

        print('Building max policy network')

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

        self.max_policy = Net(self.num_of_states, self.num_of_action_values[0]).to(self.device)

        max_policy_file = Path(self.max_policy_filename)

        if max_policy_file.is_file():
            # Load value network
            print('Loading max policy network from file ' + self.max_policy_filename)
            self.max_policy.load_state_dict(torch.load(self.max_policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        self.max_policy_criterion = torch.nn.CrossEntropyLoss()
        # self.max_policy_optimizer = optim.SGD(self.max_policy.parameters(), lr=self.max_policy_learn_rate, momentum=0)
        self.max_policy_optimizer = optim.Adam(self.max_policy.parameters(), lr=self.policy_learn_rate)

        pass

    def save_networks(self):

        torch.save(self.value.state_dict(), self.value_filename)
        torch.save(self.policy.state_dict(), self.policy_filename)
        torch.save(self.max_policy.state_dict(), self.max_policy_filename)

        pass

    # ------------------------- Memory Buffer -------------------------

    def create_memory_buffer(self):

        self.experience_mem_index = 0
        self.memory_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_states])
        self.memory_action = np.zeros([self.max_size_of_memory_buffer, self.num_of_actions])
        self.memory_reward = np.zeros([self.max_size_of_memory_buffer, 1])
        self.memory_next_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_states])
        self.memory_done = np.zeros([self.max_size_of_memory_buffer, 1])
        self.memory_validation = np.zeros([self.max_size_of_memory_buffer], dtype='bool_')

        #self.experience_mem_index = 0
        #self.memory_state = torch.empty([self.max_size_of_memory_buffer, self.num_of_states]).to(self.device)
        #self.memory_action = torch.empty([self.max_size_of_memory_buffer, self.num_of_actions]).to(self.device)
        #self.memory_reward = torch.empty([self.max_size_of_memory_buffer, 1]).to(self.device)
        #self.memory_next_state = torch.empty([self.max_size_of_memory_buffer, self.num_of_states]).to(self.device)
        #self.memory_done = torch.empty([self.max_size_of_memory_buffer, 1]).to(self.device)
        #self.memory_validation = torch.empty([self.max_size_of_memory_buffer], dtype='bool_').to(self.device)

    def load_memory_buffer(self):

        experience_buffer_file = Path(self.memory_buffer_filename)

        if experience_buffer_file.is_file():
            # Load experience buffer
            print('Loading experience buffer from file ' + self.memory_buffer_filename)
            experience_buffer = np.load(self.memory_buffer_filename)

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
            self.memory_validation[0:temp_index] = experience_buffer['validation']

        else:

            print('No experience buffer to load')

        pass

    def save_memory_buffer(self):

        state, action, reward, next_state, done, validation = self.recall_memory()

        path_without_suffix = Path(self.memory_buffer_filename).with_suffix('')

        np.savez(path_without_suffix, state=state, action=action, reward=reward, next_state=next_state, done=done,
                 validation=validation)

        pass

    def save_memory(self, state, action, reward, next_state, done):

        if self.experience_mem_index >= self.max_size_of_memory_buffer:
            self.memory_state[:-1, :] = self.memory_state[1:, :]
            self.memory_action[:-1, :] = self.memory_action[1:, :]
            self.memory_reward[:-1, :] = self.memory_reward[1:, :]
            self.memory_next_state[:-1, :] = self.memory_next_state[1:, :]
            self.memory_done[:-1, :] = self.memory_done[1:, :]
            self.memory_validation[:-1] = self.memory_validation[1:]
            self.experience_mem_index -= 1

        index = self.experience_mem_index
        self.memory_state[index, :] = state
        self.memory_action[index, :] = action
        self.memory_reward[index, :] = reward
        self.memory_next_state[index, :] = next_state
        self.memory_done[index, :] = done
        self.memory_validation[index] = np.random.choice([True, False], p=[0.3, 0.7])
        self.experience_mem_index += 1

        pass

    def recall_memory(self):

        index = self.experience_mem_index
        state = self.memory_state[0:index, :]
        action = self.memory_action[0:index, :]
        reward = self.memory_reward[0:index, :]
        next_state = self.memory_next_state[0:index, :]
        done = self.memory_done[0:index, :]
        validation = self.memory_validation[0:index]

        return state, action, reward, next_state, done, validation

    # ------------------------- Normalization -------------------------

    def scale(self, input, input_min, input_max, output_min, output_max):

        input_scaled = (input - input_min) / (input_max - input_min)
        output = input_scaled * (output_max - output_min) + output_min

        return output