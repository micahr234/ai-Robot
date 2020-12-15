import time
import copy
import numpy as np
from pathlib import Path
from collections import deque
import torch
from lib.memory import memory
import cProfile
import os
import queue
import threading


class actor_thread(threading.Thread):

    def __init__(self,
            thread_num,
            device,
            global_memory,
            environment,
            global_agents,
            transforms,
            tensor_board,
            memory_lock,
            agent_lock,
            log_lock,
            sync_barrier,
            steps,
            epochs,
            action_random_prob):

        threading.Thread.__init__(self)

        self.number = thread_num
        self.name = "ActorThread" + str(self.number)
        self.device = device
        self.global_memory = global_memory
        self.environment = environment
        self.global_agents = global_agents
        self.transforms = transforms
        self.tensor_board = tensor_board
        self.memory_lock = memory_lock
        self.agent_lock = agent_lock
        self.log_lock = log_lock
        self.sync_barrier = sync_barrier
        self.steps = steps
        self.epochs = epochs
        self.action_random_prob = action_random_prob

        self.timestep = 1
        self.reset_flag = True
        self.observation_next = None
        self.episode_timestep = None
        self.learner_agent_num = 0

        self.agent_lock.acquire()
        self.agent = copy.deepcopy(self.global_agents[self.learner_agent_num])
        self.agent_lock.release()
        self.agent.set_device(self.device)

        self.memory = memory(self.steps, None, self.device)

    def run(self):

        self.log_lock.acquire()
        print("Starting " + self.name)
        self.log_lock.release()

        for e in range(self.epochs):

            self.agent_lock.acquire()
            self.agent = copy.deepcopy(self.global_agents[self.learner_agent_num])
            self.agent_lock.release()
            self.agent.set_device(self.device)

            for t in range(self.steps):
                self.execute()
                self.timestep += 1

            self.memory_lock.acquire()
            self.global_memory.concat(self.memory)
            self.memory_lock.release()
            self.memory.clear()

            self.log_lock.acquire()
            print(self.name + " at barrier")
            self.log_lock.release()

            self.sync_barrier.wait()

        self.log_lock.acquire()
        print("Exiting " + self.name)
        self.log_lock.release()

    def execute(self):

        with torch.no_grad():

            if self.reset_flag:

                self.reset()

            self.step()

            if self.reset_flag:

                self.log_lock.acquire()
                print('---' + self.name + ' Episode Complete---\t\t\t\tEpisode timestep: ' + str(self.episode_timestep) +
                      '\t\t\t\tCumulative reward: ' + str(self.episode_cumulative_reward))
                self.tensor_board.add_scalar('record/cumulative_reward_' + self.name, self.episode_cumulative_reward, self.timestep)
                #for n in range(self.log_observation_direct_buffer.shape[1]):
                #    self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_direct_buffer[:, n], self.timestep)
                #for n in range(self.log_observation_indirect_buffer.shape[1]):
                #    self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_indirect_buffer[:, n], self.timestep)  # these may be images
                #for n in range(self.log_observation_indirect_target_buffer.shape[1]):
                #    self.tensor_board.add_histogram('observation_param' + str(n), self.log_observation_indirect_target_buffer[:, n], self.timestep)
                #for n in range(self.log_action_buffer.shape[1]):
                #    self.tensor_board.add_histogram('action_param' + str(n), self.log_action_buffer[:, n], self.timestep)
                #self.tensor_board.add_histogram('reward', torch.Tensor(self.log_reward_buffer), self.timestep)
                #self.tensor_board.add_histogram('survive', torch.Tensor(self.log_survive_buffer), self.timestep)
                self.log_lock.release()

    def reset(self):

        self.episode_timestep = 0

        observation_next_unxform = self.environment.reset()

        observation_next_direct, observation_next_indirect, observation_next_indirect_target = self.transforms["observation_input"](observation_next_unxform, self.episode_timestep)
        observation_next_direct = observation_next_direct.to(self.device)
        observation_next_indirect = observation_next_indirect.to(self.device)
        observation_next_indirect_target = observation_next_indirect_target.to(self.device)

        self.episode_cumulative_reward = 0

        self.observation_next = (observation_next_direct, observation_next_indirect, observation_next_indirect_target)

        self.environment.render()

    def step(self):

        self.episode_timestep += 1

        observation_direct, observation_indirect, observation_indirect_target = self.observation_next

        action = self.agent.act(self.action_random_prob, observation_direct=observation_direct, observation_indirect=observation_indirect).squeeze(0)

        action_limited = torch.clamp(action, -1.0, 1.0)
        assert torch.all(action == action_limited)

        action_xform = self.transforms["action_output"](action)
        observation_next_unxform, reward_unxform, terminate_unxform, info = self.environment.step(action_xform)
        survive_unxform = 1.0 - terminate_unxform

        observation_next_direct, observation_next_indirect, observation_next_indirect_target = self.transforms["observation_input"](observation_next_unxform, self.episode_timestep)
        observation_next_direct = observation_next_direct.to(self.device)
        observation_next_indirect = observation_next_indirect.to(self.device)
        observation_next_indirect_target = observation_next_indirect_target.to(self.device)
        reward = self.transforms["reward_input"](reward_unxform).to(self.device)
        survive = self.transforms["survive_input"](survive_unxform).to(self.device)

        self.episode_cumulative_reward += reward.item()

        self.observation_next = (observation_next_direct, observation_next_indirect, observation_next_indirect_target)

        self.memory.add(
            observation_direct=observation_direct,
            observation_indirect=observation_indirect,
            observation_indirect_target=observation_indirect_target,
            action=action,
            reward=reward,
            survive=survive,
            observation_next_direct=observation_next_direct,
            observation_next_indirect=observation_next_indirect,
            observation_next_indirect_target=observation_next_indirect_target
        )

        self.environment.render()

        self.reset_flag = False if survive == 1.0 else True
