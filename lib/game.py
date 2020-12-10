import time
import copy
import numpy as np
from pathlib import Path
from collections import deque
import torch
from lib.memory import memory
import cProfile
import os


class game():

    def __init__(self,
            name,
            environment,
            agent,
            tensor_board,
            learn_interval,
            max_timestep,
            memory_buffer_size,
            max_episode_timestamp,
            batch_size,
            batches,
            render_delay,
            save,
            profile,
            observation_input_transform,
            action_input_transform,
            reward_input_transform,
            survive_input_transform,
            action_output_transform,
            ):

        print('')
        print('Creating Game')

        self.name = name
        self.environment = environment
        self.agent = agent
        self.tensor_board = tensor_board
        self.learn_interval = learn_interval
        self.max_timestep = max_timestep
        self.memory_buffer_size = memory_buffer_size
        self.max_episode_timestamp = max_episode_timestamp
        self.batch_size = batch_size
        self.batches = batches
        self.render_delay = render_delay
        self.save = save
        self.profile = profile

        self.observation_input_transform = observation_input_transform
        self.action_input_transform = action_input_transform
        self.reward_input_transform = reward_input_transform
        self.survive_input_transform = survive_input_transform
        self.action_output_transform = action_output_transform

        self.device = torch.device('cpu')

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        self.memory_filename = self.memory_dir / 'memory.pt'
        self.memory = memory(self.memory_buffer_size, self.memory_filename, self.device)

        self.action_space_min = torch.FloatTensor(self.environment.action_space.low, device=self.device)
        self.action_space_max = torch.FloatTensor(self.environment.action_space.high, device=self.device)

        self.observation_next = None

        self.log_observation_direct_buffer = deque()
        self.log_observation_indirect_buffer = deque()
        self.log_observation_indirect_target_buffer = deque()
        self.log_action_buffer = deque()
        self.log_reward_buffer = deque()
        self.log_survive_buffer = deque()

        self.learnstep = None
        self.timestep = None
        self.learn_flag = None
        self.reset_flag = True


    def run(self):

        print('')
        print('Starting Game')

        if self.profile:
            pr = cProfile.Profile()
            pr.enable()

        self.learnstep = 1

        for t in range(1, self.max_timestep + 1):

            self.timestep = t

            if self.reset_flag:

                with torch.no_grad():
                    self.reset()

                if self.render_delay > 0.0:
                    time.sleep(self.render_delay)

            with torch.no_grad():
                self.step()

            if self.render_delay > 0.0:
                time.sleep(self.render_delay)

            if self.reset_flag:

                print('---Episode Complete---\t\t\t\tEpisode timestep: ' + str(self.episode_timestep) +
                      '\t\t\t\tCumulative reward: ' + str(self.episode_cumulative_reward))

                self.log_reward()
                #self.log_values()

            self.learn_flag = (self.timestep / self.learn_interval) >= self.learnstep

            if self.learn_flag:

                self.learnstep += 1
                data = self.memory.get_all()
                self.agent.learn(data, self.batch_size, self.batches)

                data_length = data['survive'].shape[0]
                print('***Learing Complete***\t\t\t\tSamples: ' + str(data_length) +
                      '\t\t\t\tBatch Size: ' + str(self.batch_size) +
                      '\t\t\t\tNumber Of Batches: ' + str(self.batches))

                if self.save:

                    self.agent.save()
                    self.memory.save()

        if self.save:

            self.agent.save()
            self.memory.save()

        if self.profile:
            pr.disable()
            profile_dir = Path.cwd() / 'profile' / self.name
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            profile_filename = profile_dir / 'profile.pt'
            pr.dump_stats(profile_filename)
            print('Follow the link below to see the time profile')
            os.system("snakeviz " + str(profile_filename))


    def reset(self):

        self.episode_timestep = 0

        observation_next_unxform = self.environment.reset()

        observation_next_direct, observation_next_indirect, observation_next_indirect_target = self.observation_input_transform(observation_next_unxform)
        observation_next_direct = observation_next_direct.to(self.device)
        observation_next_indirect = observation_next_indirect.to(self.device)
        observation_next_indirect_target = observation_next_indirect_target.to(self.device)
        if self.max_episode_timestamp is not None:
            observation_next_direct = torch.cat((observation_next_direct, (torch.FloatTensor([self.episode_timestep]) / self.max_episode_timestamp) * 2.0 - 1.0), dim=-1)

        self.episode_cumulative_reward = 0

        self.observation_next = (observation_next_direct, observation_next_indirect, observation_next_indirect_target)

        self.environment.render()

    def step(self):

        self.episode_timestep += 1

        observation_direct, observation_indirect, observation_indirect_target = self.observation_next

        action = self.agent.act(observation_direct=observation_direct, observation_indirect=observation_indirect).squeeze(0).to(self.device)

        action_limited = torch.clamp(action, -1.0, 1.0)
        assert torch.all(action == action_limited)

        scaled_action = self.scale_action(action)
        action_xform = self.action_output_transform(scaled_action)
        observation_next_unxform, reward_unxform, terminate_unxform, info = self.environment.step(action_xform)
        survive_unxform = 1.0 - terminate_unxform

        observation_next_direct, observation_next_indirect, observation_next_indirect_target = self.observation_input_transform(observation_next_unxform)
        observation_next_direct = observation_next_direct.to(self.device)
        observation_next_indirect = observation_next_indirect.to(self.device)
        observation_next_indirect_target = observation_next_indirect_target.to(self.device)
        if self.max_episode_timestamp is not None:
            observation_next_direct = torch.cat((observation_next_direct, (torch.FloatTensor([self.episode_timestep]) / self.max_episode_timestamp) * 2.0 - 1.0), dim=-1)
        reward = self.reward_input_transform(reward_unxform).to(self.device)
        survive = self.survive_input_transform(survive_unxform).to(self.device)

        self.episode_cumulative_reward += reward.item()

        self.observation_next = (observation_next_direct, observation_next_indirect, observation_next_indirect_target)

        self.memory.add(
            observation_direct=observation_direct,
            observation_indirect=observation_indirect,
            observation_indirect_target=observation_indirect_target,
            action=action,
            reward=reward,
            observation_next_direct=observation_next_direct,
            observation_next_indirect=observation_next_indirect,
            observation_next_indirect_target=observation_next_indirect_target,
            survive=survive
        )

        #self.log_observation_direct_buffer.append(observation_direct_buffer[0])
        #self.log_observation_indirect_buffer.append(observation_indirect_buffer[0])
        #self.log_observation_indirect_target_buffer.append(observation_indirect_target_buffer[0])
        #self.log_action_buffer.append(action)
        #self.log_reward_buffer.append(reward)
        #self.log_survive_buffer.append(survive)

        self.environment.render()

        self.reset_flag = False if survive == 1.0 else True


    def scale_action(self, action):

        scaled_action = (action / 2 + 0.5) * (self.action_space_max - self.action_space_min) + self.action_space_min

        return scaled_action


    def log_reward(self):

        self.tensor_board.add_scalar('record/cumulative_reward', self.episode_cumulative_reward, self.timestep)


    def log_values(self):

        for n in range(self.log_observation_direct_buffer.shape[1]):
            self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_direct_buffer[:, n], self.timestep)
        for n in range(self.log_observation_indirect_buffer.shape[1]):
            self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_indirect_buffer[:, n], self.timestep) #these may be images
        for n in range(self.log_observation_indirect_target_buffer.shape[1]):
            self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_indirect_target_buffer[:, n], self.timestep)
        for n in range(self.log_action_buffer.shape[1]):
            self.tensor_board.add_histogram('action_param' + str(n), self.log_action_buffer[:, n], self.timestep)
        self.tensor_board.add_histogram('reward', torch.Tensor(self.log_reward_buffer), self.timestep)
        self.tensor_board.add_histogram('survive', torch.Tensor(self.log_survive_buffer), self.timestep)

        self.log_observation_direct_buffer.clear()
        self.log_observation_indirect_buffer.clear()
        self.log_observation_indirect_target_buffer.clear()
        self.log_action_buffer.clear()
        self.log_reward_buffer.clear()
        self.log_survive_buffer.clear()
