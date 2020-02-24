import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from experienceMemory import ExperienceMemory
from torch.utils.tensorboard import SummaryWriter


# Define reward prediction network
class AgentTorchDiscrete():

    # ------------------------- Initialization -------------------------

    def __init__(self, name, num_of_action_values, state_space_min, state_space_max, reward_space_min, reward_space_max,
                 batch_size=1000, epoch_size=10000, learn_iterations=10, memory_buffer_size=100000,
                 discount=0.999, value_learn_rate=0.0001, policy_learn_rate=0.00001, policy_copy_rate=0.0001, next_learn_factor=0.5,
                 prioritize_weight_exponent=2, prioritize_weight_min=0.5, prioritize_weight_max=5.0, prioritize_weight_copy_rate=0.01,
                 debug=False):

        self.debug = debug

        self.name = str(name)

        print('Creating agent ' + self.name)

        self.memory_dir_path = Path.cwd() / 'memory' / name
        self.value_filename = self.memory_dir_path / 'value.pt'
        self.policy_filename = self.memory_dir_path / 'policy.pt'
        self.max_policy_filename = self.memory_dir_path / 'max_policy.pt'
        self.memory_buffer_filename = self.memory_dir_path / 'data.pt'

        Path(self.memory_dir_path).mkdir(parents=True, exist_ok=True)

        self.discount = discount
        self.value_learn_rate = value_learn_rate
        self.policy_copy_rate = policy_copy_rate
        self.policy_learn_rate = policy_learn_rate
        self.learn_iterations = learn_iterations
        self.max_size_of_memory_buffer = memory_buffer_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.next_learn_factor = next_learn_factor
        self.prioritize_weight_exponent = prioritize_weight_exponent
        self.prioritize_weight_min = prioritize_weight_min
        self.prioritize_weight_max = prioritize_weight_max
        self.prioritize_weight_copy_rate = prioritize_weight_copy_rate

        # dim1 = variables
        # example [5, 3, 3]
        self.num_of_action_values = num_of_action_values
        self.num_of_actions = len(self.num_of_action_values)

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

        self.memory = ExperienceMemory(self.max_size_of_memory_buffer, self.num_of_states, self.memory_buffer_filename,
                                       weight_initialization=self.prioritize_weight_max, weight_exponent=self.prioritize_weight_exponent)

        self.tensor_board = SummaryWriter('runs/' + self.name)

        self.epoch_count = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state, use_max_policy=False):

        in_state = np.array(in_state, ndmin=2)
        in_state = self.scale(in_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        state = torch.from_numpy(in_state).float().detach().to(self.device)

        self.policy.eval()
        self.max_policy.eval()

        with torch.no_grad():
            if use_max_policy:
                policy_logits = self.max_policy.forward(state)[0]
            else:
                policy_logits = self.policy.forward(state)[0]

            policy_probs_flat = F.softmax(policy_logits, dim=-1)
            action_flat = torch.multinomial(policy_probs_flat, 1, replacement=True)

        out_action = action_flat.cpu().numpy()
        out_action = self.action_unflatten(out_action)

        return out_action

    def record(self, in_state, in_action, in_reward, in_next_state, in_done, in_timestep):

        in_state = np.array(in_state, ndmin=2)
        in_state = self.scale(in_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_action = np.array(in_action, ndmin=1)
        in_action = self.action_flatten(in_action)
        in_action = np.array(in_action, ndmin=2)
        in_reward = np.array(in_reward, ndmin=2)
        in_reward = self.scale(in_reward, self.reward_space_min_array, self.reward_space_max_array, -1, 1)
        in_next_state = np.array(in_next_state, ndmin=2)
        in_next_state = self.scale(in_next_state, self.state_space_min_array, self.state_space_max_array, -1, 1)
        in_done = np.array(in_done, ndmin=2)

        self.memory.add((in_state, in_action, in_reward, in_next_state, in_done), self.prioritize_weight_max)

        pass

    def learn(self):

        print('Agent ' + str(self.name) + ' learning fom ' + str(self.memory.len()) + ' samples')

        if self.debug: start_time = time.time(); elapsed_time = time.time() - start_time; print('Started at: ' + str(elapsed_time))

        dataset_index = self.memory.prepare_dataset(min(self.epoch_size, self.memory.len()), True)

        for epoch in range(self.learn_iterations):

            if self.debug: elapsed_time = time.time() - start_time; print('Begin epoch: ' + str(elapsed_time))

            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0
            running_count = 0.0
            batch_num = 1

            while True:

                index, state, action, reward, next_state, done, last_batch = self.memory.get_batch(dataset_index, self.batch_size, batch_num)

                if self.debug: elapsed_time = time.time() - start_time; print('Begin batch: ' + str(elapsed_time))

                # set the model to train mode
                self.value.train()
                self.max_policy.train()

                # forward pass
                if self.debug: elapsed_time = time.time() - start_time; print('Begin forward: ' + str(elapsed_time))
                values = self.value(state)
                values_sum = torch.gather(values, 1, action.long())

                max_policy_logits = self.max_policy.forward(next_state)
                max_policy_probs = F.softmax(max_policy_logits.detach(), dim=1)
                values_next_with_grad = self.value(next_state)
                values_next = self.next_learn_factor * values_next_with_grad \
                              + (1.0 - self.next_learn_factor) * values_next_with_grad.detach()
                values_next_sum = torch.sum(values_next * max_policy_probs, 1, keepdim=True)

                values_diff = values_sum - values_next_sum * self.discount * (1.0 - done)
                policy_ground_truth = torch.argmax(values_next_with_grad.detach(), dim=1)

                # optimize value
                if self.debug: elapsed_time = time.time() - start_time; print('Begin opto value: ' + str(elapsed_time))
                self.value_optimizer.zero_grad()
                value_loss = self.value_criterion(values_diff, reward)
                value_loss.backward(retain_graph=True)
                self.value_optimizer.step()

                # optimize max policy
                if self.debug: elapsed_time = time.time() - start_time; print('Begin opto policy: ' + str(elapsed_time))
                self.max_policy_optimizer.zero_grad()
                policy_loss = self.max_policy_criterion(max_policy_logits, policy_ground_truth)
                policy_loss.backward()
                self.max_policy_optimizer.step()

                # copy policy
                if self.debug: elapsed_time = time.time() - start_time; print('Begin copy policy: ' + str(elapsed_time))
                policy_dict = self.policy.state_dict()
                max_policy_dict = self.max_policy.state_dict()
                for param_name in self.policy.state_dict():
                    policy_dict[param_name] = (1 - self.policy_copy_rate) * policy_dict[
                        param_name] + self.policy_copy_rate * max_policy_dict[param_name]
                    pass
                self.policy.load_state_dict(policy_dict)

                # update weights
                if self.debug: elapsed_time = time.time() - start_time; print('Update weights: ' + str(elapsed_time))
                train_weights = abs(reward - values_diff.detach()) * abs(self.reward_space_max_array - self.reward_space_min_array)
                train_weights_clamped = torch.clamp(train_weights, min=self.prioritize_weight_min, max=self.prioritize_weight_max)
                self.memory.set_weight(index, train_weights_clamped.float(), self.prioritize_weight_copy_rate)

                # gather statistics
                if self.debug: elapsed_time = time.time() - start_time; print('Gather stats: ' + str(elapsed_time))
                epoch_value_loss += value_loss.item()
                epoch_policy_loss += policy_loss.item()
                running_count += 1.0

                if self.debug: elapsed_time = time.time() - start_time; print('End batch: ' + str(elapsed_time))

                # check if done
                if last_batch:
                    break
                else:
                    batch_num += 1

            self.epoch_count += 1

            if self.debug: elapsed_time = time.time() - start_time; print('End epoch: ' + str(elapsed_time))

            print('Epoch: ' + str(self.epoch_count)
                  + ' \t\tvalue loss:' + str(epoch_value_loss / running_count)
                  + ' \t\tpolicy loss:' + str(epoch_policy_loss / running_count))

            self.tensor_board.add_scalar('Loss/value', epoch_value_loss / running_count, self.epoch_count)
            self.tensor_board.add_scalar('Loss/policy', epoch_policy_loss / running_count, self.epoch_count)

    pass

    def save(self):

        print('Saving network and experience')
        self.save_networks()
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_value_network(self):

        print('Building value network')

        class Net(nn.Module):

            def __init__(self, num_of_states, num_of_action_values):
                super(Net, self).__init__()
                self.num_of_action_values = num_of_action_values
                self.fc1 = nn.Linear(num_of_states, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, np.prod(self.num_of_action_values))

            def forward(self, state_input):
                x = F.relu(self.fc1(state_input))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.value = Net(self.num_of_states, self.num_of_action_values).to(self.device)

        value_file = Path(self.value_filename)

        if value_file.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_criterion = nn.MSELoss()
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.value_learn_rate)

        pass

    def build_policy_network(self):

        print('Building policy network')

        class Net(nn.Module):

            def __init__(self, num_of_states, num_of_action_values):
                super(Net, self).__init__()
                self.num_of_action_values = num_of_action_values
                self.fc1 = nn.Linear(num_of_states, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 64)
                self.fc4 = nn.Linear(64, 32)
                self.fc5 = nn.Linear(32, np.prod(self.num_of_action_values))

            def forward(self, state_input):
                x = F.relu(self.fc1(state_input))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
                x = self.fc5(x)
                return x

        self.policy = Net(self.num_of_states, self.num_of_action_values).to(self.device)

        policy_file = Path(self.policy_filename)

        if policy_file.is_file():
            # Load value network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build value network
            print('No policy network loaded from file')

        self.max_policy = Net(self.num_of_states, self.num_of_action_values).to(self.device)

        max_policy_file = Path(self.max_policy_filename)

        if max_policy_file.is_file():
            # Load value network
            print('Loading max policy network from file ' + str(self.max_policy_filename))
            self.max_policy.load_state_dict(torch.load(self.max_policy_filename))

        else:
            # Build value network
            print('No max policy network loaded from file')

        self.max_policy_criterion = torch.nn.CrossEntropyLoss()
        self.max_policy_optimizer = optim.Adam(self.max_policy.parameters(), lr=self.policy_learn_rate)

        pass

    def save_networks(self):

        torch.save(self.value.state_dict(), self.value_filename)
        torch.save(self.policy.state_dict(), self.policy_filename)
        torch.save(self.max_policy.state_dict(), self.max_policy_filename)

        pass


    # ------------------------- Normalization -------------------------

    def scale(self, input, input_min, input_max, output_min, output_max):

        input_scaled = (input - input_min) / (input_max - input_min)
        output = input_scaled * (output_max - output_min) + output_min

        return output

    def action_flatten(self, in_action):
        num_flat_actions = np.prod(self.num_of_action_values)
        flat_action_array = np.arange(num_flat_actions)
        action_array = np.reshape(flat_action_array, self.num_of_action_values)
        a = action_array
        for n in in_action:
            a = a[n]
        out_action = np.array(a)
        return out_action

    def action_unflatten(self, in_action):
        num_flat_actions = np.prod(self.num_of_action_values)
        flat_action_array = np.arange(num_flat_actions)
        action_array = np.reshape(flat_action_array, self.num_of_action_values)
        out_action = np.where(action_array == in_action)
        out_action = np.array([n[0] for n in out_action])
        return out_action