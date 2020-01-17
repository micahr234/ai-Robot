import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt


# Define reward prediction network
class Agent():


    # ------------------------- Initialization -------------------------

    def __init__(self, env, name, action_temperature=float('inf'), action_temperature_multiplier=1, discount=0.9999, batch_size=1000, learn_rate=0.0001, memory_buffer_size=100000, next_learn_factor=0.1, num_of_hypothetical_actions=1000):

        self.name = name

        print('Creating agent ' + str(self.name))

        self.q_value_filename = name + '.h5'
        self.memory_buffer_filename = name + '.npz'

        self.discount = discount
        self.network_learn_rate = learn_rate
        self.action_temperature = action_temperature
        self.action_temperature_multiplier = action_temperature_multiplier
        self.max_size_of_memory_buffer = memory_buffer_size
        self.batch_size = batch_size
        self.next_learn_factor = next_learn_factor
        self.num_of_hypothetical_actions = num_of_hypothetical_actions

        self.num_of_action_variables = env.action_space.shape[0]
        self.num_of_state_variables = env.observation_space.shape[0]
        self.action_min = env.action_space.low
        self.action_max = env.action_space.high
        self.reward_min = env.reward_range[0] if (env.reward_range[0] != -np.inf) else 0 # change back to -1
        self.reward_max = env.reward_range[1] if (env.reward_range[1] != np.inf) else 14 # change back to 1
        self.state_min = env.observation_space.low
        self.state_max = env.observation_space.high

        self.build_network()
        self.build_learner_network()

        self.create_memory_buffer()
        self.load_memory_buffer()
        self.new_memories = 0


    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        [state] = self.preprocess(state=in_state)

        action = self.find_action(state, temperature=self.action_temperature)

        [out_action] = self.postprocess(action=action)

        return out_action[0]

    def react(self, in_state, in_action, in_reward, in_next_state, in_done):

        [state, action, reward, next_state, done] = self.preprocess(state=in_state, action=in_action, reward=in_reward, next_state=in_next_state, done=in_done)
        self.save_memory(state, action, reward, next_state, done)
        self.new_memories += 1

        if self.new_memories >= self.batch_size:
            self.train()
            self.new_memories = 0
            self.save_networks()
            self.save_memory_buffer()

        self.action_temperature = self.action_temperature * self.action_temperature_multiplier

        pass


    # ------------------------- Sub Functions -------------------------

    def train(self):

        print('Agent ' + str(self.name) + ' learing')

        state, action, reward, next_state, done, validation = self.recall_memory()

        max_next_action = self.find_action(next_state)

        self.q_value_learner.fit([state, action, next_state, max_next_action, done], [reward], verbose=2, epochs=10, batch_size=self.batch_size, shuffle=True)

        pass

    def find_action(self, in_states, temperature=np.inf):

        num_of_states = in_states.shape[0]

        repeated_states = np.repeat(in_states, self.num_of_hypothetical_actions, axis=0)

        actions = self.generate_actions(number=num_of_states*self.num_of_hypothetical_actions)

        q_values = self.q_value.predict([repeated_states, actions])

        max_actions = np.zeros((num_of_states, self.num_of_action_variables))
        for i in range(num_of_states):

            index = i * self.num_of_hypothetical_actions + np.arange(self.num_of_hypothetical_actions)
            index = index.flatten()
            hypothetical_actions = actions[index, :]
            hypothetical_q_values = q_values[index, :]

            if temperature < float('inf'):
                action_probs = self.softmax(hypothetical_q_values, temperature)
                action_index = np.random.choice(range(action_probs.shape[0]), p=action_probs.squeeze(axis=1))
            else:
                action_index = np.argmax(hypothetical_q_values)

            max_actions[i, :] = hypothetical_actions[action_index, :]

        return max_actions

    def softmax(self, x, temperature):

        x_new = x * temperature
        e_x = np.exp(x_new - np.max(x_new))

        return e_x / e_x.sum()

    def generate_actions(self, number=1):

        actions = np.random.rand(number, self.num_of_action_variables)

        return actions


    # ------------------------- Network Functions -------------------------

    def build_network(self):

        q_value_file = Path(self.q_value_filename)

        if q_value_file.is_file() or q_value_file.is_dir():
            # Load value network
            print('Loading network from file ' + self.q_value_filename)
            self.q_value = keras.models.load_model(self.q_value_filename)

        else:
            # Build value network
            print('Building network')

            state_input = keras.layers.Input(shape=(self.num_of_state_variables,))
            action_input = keras.layers.Input(shape=(self.num_of_action_variables,))

            q_value_b1 = keras.layers.Concatenate()([state_input, action_input])
            q_value_b1 = keras.layers.Dense(256, activation='relu')(q_value_b1)
            q_value_b1 = keras.layers.Dense(128, activation='relu')(q_value_b1)
            q_value_b1 = keras.layers.Dense(64, activation='relu')(q_value_b1)
            q_value_b1 = keras.layers.Dense(32, activation='relu')(q_value_b1)
            value_output = keras.layers.Dense(1, activation='linear')(q_value_b1)

            self.q_value = keras.models.Model([state_input, action_input], [value_output], name='q_value')

        self.q_value.trainable = True
        self.q_value.compile(optimizer=keras.optimizers.Adam(lr=self.network_learn_rate), loss='mse')
        # self.q_value.summary()

        pass

    def build_learner_network(self):

        # Build value network
        print('Building learner network')

        state_input = keras.layers.Input(shape=(self.num_of_state_variables,))
        action_input = keras.layers.Input(shape=(self.num_of_action_variables,))
        next_state_input = keras.layers.Input(shape=(self.num_of_state_variables,))
        max_next_action_input = keras.layers.Input(shape=(self.num_of_action_variables,))
        done_input = keras.layers.Input(shape=(1,))

        q_value = self.q_value([state_input, action_input])
        next_q_value = self.q_value([next_state_input, max_next_action_input])
        next_q_value_mod = keras.layers.Lambda(lambda x: self.next_learn_factor * x + tf.stop_gradient((1 - self.next_learn_factor) * x))(next_q_value)
        reward_out = keras.layers.Lambda(lambda x: x[0] - x[1] * self.discount * (1.0 - x[2]))([q_value, next_q_value_mod, done_input])

        self.q_value_learner = keras.models.Model([state_input, action_input, next_state_input, max_next_action_input, done_input], [reward_out], name='q_value_learner')

        self.q_value_learner.trainable = True
        self.q_value_learner.compile(optimizer=keras.optimizers.Adam(lr=self.network_learn_rate), loss='mse')
        # self.q_value_learner.summary()

        pass

    def save_networks(self):

        self.q_value.save(self.q_value_filename)

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
            raise ValueError('Experience memory buffer overflow.')

        state = self.normalize_state(state)
        action = self.normalize_action(action)
        reward = self.normalize_reward(reward)
        next_state = self.normalize_state(next_state)

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