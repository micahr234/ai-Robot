import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define reward prediction network
class AgentTorch():


    # ------------------------- Initialization -------------------------

    def __init__(self, env, name, discount=0.9999, batch_size=1000, learn_rate=0.0001, learn_iterations=10, memory_buffer_size=0, next_learn_factor=0.1, num_of_hypothetical_actions=1000):

        self.name = name

        print('Creating agent ' + str(self.name))

        self.q_value_filename = name + '.pt'
        self.memory_buffer_filename = name + '.npz'

        self.discount = discount
        self.learn_rate = learn_rate
        self.learn_iterations = learn_iterations
        self.max_size_of_memory_buffer = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor
        self.num_of_hypothetical_actions = num_of_hypothetical_actions

        self.num_of_action_variables = env.action_space.shape[0]
        self.num_of_state_variables = env.observation_space.shape[0]
        self.action_min = env.action_space.low
        self.action_max = env.action_space.high
        self.reward_min = -1
        self.reward_max = 1
        #self.reward_min = env.reward_range[0] if (env.reward_range[0] != -np.inf) else -16.2736044 # change back to -1
        #self.reward_max = env.reward_range[1] if (env.reward_range[1] != np.inf) else 0 # change back to 1
        self.state_min = env.observation_space.low
        self.state_max = env.observation_space.high

        self.build_network()

        if self.max_size_of_memory_buffer > 0:
            self.create_memory_buffer()
            self.load_memory_buffer()


    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state, prob_factor=0):

        [state] = self.preprocess(state=in_state)

        action = self.find_action(state, prob_factor=prob_factor)

        [out_action] = self.postprocess(action=action)

        return out_action[0]

    def record(self, in_state, in_action, in_reward, in_next_state, in_done):

        [state, action, reward, next_state, done] = self.preprocess(state=in_state, action=in_action, reward=in_reward, next_state=in_next_state, done=in_done)
        self.save_memory(state, action, reward, next_state, done)

        pass

    def learn(self):

        self.train()
        self.save_networks()
        self.save_memory_buffer()

        pass


    # ------------------------- Sub Functions -------------------------

    def train(self):

        print('Agent ' + str(self.name) + ' learning')

        state, action, reward, next_state, done, validation = self.recall_memory()

        max_next_action = self.find_action(next_state)

        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()
        next_state = torch.from_numpy(next_state).float()
        done = torch.from_numpy(done).float()
        max_next_action = torch.from_numpy(max_next_action).float()

        trainset = torch.utils.data.TensorDataset(state, action, next_state, max_next_action, done, reward)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.learn_iterations):

            running_loss = 0.0
            running_count = 0.0
            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                in_state, in_action, in_next_state, in_max_next_action, in_done, out_reward = data

                # set the model to train mode
                self.q_value.train()

                # forward + backward + optimize
                self.optimizer.zero_grad()
                q_values = self.q_value(in_state, in_action)
                with torch.no_grad():
                    q_values_next_no_grad = self.q_value(in_next_state, in_max_next_action)
                q_values_next_with_grad = self.q_value(in_next_state, in_max_next_action)
                q_values_next = self.next_learn_factor * q_values_next_with_grad + (1.0 - self.next_learn_factor) * q_values_next_no_grad
                outputs = q_values - q_values_next * self.discount * (1.0 - in_done)
                loss = self.criterion(outputs, out_reward)
                loss.backward()
                self.optimizer.step()

                # gather statistics
                running_loss += loss.item()
                running_count += 1.0

            print('Epoch: ' + str(epoch + 1) + ' Loss:' + str(running_loss / running_count))

        pass

    def find_action(self, in_states, prob_factor=0):

        num_of_states = in_states.shape[0]

        repeated_states = np.repeat(in_states, self.num_of_hypothetical_actions, axis=0)

        actions = self.generate_actions(number=num_of_states)

        repeated_states_t = torch.from_numpy(repeated_states).float()
        actions_t = torch.from_numpy(actions).float()

        self.q_value.eval()

        with torch.no_grad():
            q_values_t = self.q_value(repeated_states_t, actions_t)

        q_values = q_values_t.numpy()

        max_actions = np.zeros((num_of_states, self.num_of_action_variables))
        for i in range(num_of_states):

            index = i * self.num_of_hypothetical_actions + np.arange(self.num_of_hypothetical_actions)
            index = index.flatten()
            hypothetical_actions = actions[index, :]
            #hypothetical_actions = np.sort(hypothetical_actions, axis=0) # For debug only
            hypothetical_q_values = q_values[index, :]

            action_probs = self.softmax(hypothetical_q_values, prob_factor)
            action_index = np.random.choice(range(action_probs.shape[0]), p=action_probs.squeeze(axis=1))

            max_actions[i, :] = hypothetical_actions[action_index, :]

        return max_actions

    def softmax(self, x, prob):

        x_avg = x / np.abs(x.sum())
        x_new = x_avg / (prob + 1e-12)
        e_x = np.exp(x_new - np.max(x_new))
        return e_x / e_x.sum()

    def generate_actions(self, number=1):

        actions = np.zeros([self.num_of_hypothetical_actions*number, self.num_of_action_variables])
        for n in range(number):
            i = n * self.num_of_hypothetical_actions + np.arange(self.num_of_hypothetical_actions)
            # actions[i, :] = np.random.uniform(-1, 1, (self.num_of_hypothetical_actions, self.num_of_action_variables))
            actions[i, :] = np.array(np.linspace(-1, 1, num=self.num_of_hypothetical_actions), ndmin=2).transpose()

        return actions


    # ------------------------- Network Functions -------------------------

    def build_network(self):

        print('Building value network')

        class Net(nn.Module):

            def __init__(self, state_in_size, action_in_size):
                super(Net, self).__init__()
                self.state_in_size = state_in_size
                self.action_in_size = action_in_size
                self.fc1 = nn.Linear(self.state_in_size + self.action_in_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, 1)

            def forward(self, state_input, action_input):
                x = torch.cat((state_input, action_input), dim=1)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.q_value = Net(self.num_of_state_variables, self.num_of_action_variables)

        q_value_file = Path(self.q_value_filename)

        if q_value_file.is_file():
            # Load value network
            print('Loading network from file ' + self.q_value_filename)
            self.q_value.load_state_dict(torch.load(self.q_value_filename))

        else:
            # Build value network
            print('No network loaded from file')

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.q_value.parameters(), lr=self.learn_rate, momentum=0)
        #self.optimizer = optim.Adam(self.q_value.parameters(), lr=self.learn_rate)

        pass


    def save_networks(self):

        torch.save(self.q_value.state_dict(), self.q_value_filename)

        pass


    # ------------------------- Preprocessing & Postprocessing -------------------------

    def preprocess(self, state=None, action=None, reward=None, next_state=None, done=None):

        output = []

        if state is not None:
            out_state = np.array(state, ndmin=2)
            out_state = self.normalize_state(out_state)
            output.append(out_state)

        if action is not None:
            out_action = np.array(action, ndmin=2)
            out_action = self.normalize_action(out_action)
            output.append(out_action)

        if reward is not None:
            out_reward = np.array(reward, ndmin=2)
            out_reward = self.normalize_reward(out_reward)
            output.append(out_reward)

        if next_state is not None:
            out_next_state = np.array(next_state, ndmin=2)
            out_next_state = self.normalize_state(out_next_state)
            output.append(out_next_state)

        if done is not None:
            out_done = np.array(done, ndmin=2)
            output.append(out_done)

        return output

    def postprocess(self, action=None):

        output = []

        if action is not None:
            out_action = self.unnormalize_action(action)
            output.append(out_action)

        return output

    def normalize_reward(self, reward):

        reward_normalized = self.scale(reward, self.reward_min, self.reward_max, np.array([-1]), np.array([1]))

        return reward_normalized

    def unnormalize_reward(self, reward):

        reward_unnormalized = self.scale(reward, np.array([-1]), np.array([1]), self.reward_min, self.reward_max)

        return reward_unnormalized

    def normalize_state(self, state):

        state_normalized = self.scale(state, self.state_min, self.state_max, np.array([-1]), np.array([1]))

        return state_normalized

    def unnormalize_state(self, state):

        state_unnormalized = self.scale(state, np.array([-1]), np.array([1]), self.state_min, self.state_max)

        return state_unnormalized

    def normalize_action(self, action):

        action_normalized = self.scale(action, self.action_min, self.action_max, np.array([-1]), np.array([1]))

        return action_normalized

    def unnormalize_action(self, action):

        action_unnormalized = self.scale(action, np.array([-1]), np.array([1]), self.action_min, self.action_max)

        return action_unnormalized

    def scale(self, input, input_min, input_max, output_min, output_max):

        input_scaled = (input - input_min) / (input_max - input_min)
        output = input_scaled * (output_max - output_min) + output_min

        return output


    # ------------------------- Memory Buffer -------------------------

    def create_memory_buffer(self):

        self.experience_mem_index = 0
        self.memory_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_state_variables])
        self.memory_action = np.zeros([self.max_size_of_memory_buffer, self.num_of_action_variables])
        self.memory_reward = np.zeros([self.max_size_of_memory_buffer, 1])
        self.memory_next_state = np.zeros([self.max_size_of_memory_buffer, self.num_of_state_variables])
        self.memory_done = np.zeros([self.max_size_of_memory_buffer, 1])
        self.memory_validation = np.zeros([self.max_size_of_memory_buffer], dtype='bool_')

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

        np.savez(path_without_suffix, state=state, action=action, reward=reward, next_state=next_state, done=done, validation=validation)

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