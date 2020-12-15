import time
import copy
import numpy as np
from pathlib import Path
from collections import deque
import torch
from lib.memory import memory
from lib.actor_thread import actor_thread
from lib.learner_thread import learner_thread
import cProfile
import os
import queue
import threading


class game():

    def __init__(self,
            name,
            environment,
            agent,
            tensor_board,
            epochs,
            steps,
            memory_buffer_size,
            batch_size,
            save,
            profile,
            observation_input_transform,
            reward_input_transform,
            survive_input_transform,
            action_output_transform,
            action_random_prob):

        print('')
        print('Creating Game')

        self.name = name
        self.environment = environment
        self.agent = agent
        self.tensor_board = tensor_board
        self.epochs = epochs
        self.steps = steps
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.save = save
        self.profile = profile
        self.action_random_prob = action_random_prob

        self.transforms = {"observation_input": observation_input_transform,
                           "reward_input": reward_input_transform,
                           "survive_input": survive_input_transform,
                           "action_output": action_output_transform}

        self.actor_device = torch.device('cuda:1')
        self.learner_device = torch.device('cuda:0')

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)
        self.memory_filename = self.memory_dir / 'memory.pt'
        self.memory = memory(self.memory_buffer_size, self.memory_filename, self.actor_device)

        self.num_of_actor_threads = len(environment)
        self.num_of_learner_threads = 1

    def run(self):

        print('')
        print('Starting Game')

        if self.profile:

            pr = cProfile.Profile()
            pr.enable()

        memory_lock = threading.Lock()
        agent_lock = threading.Lock()
        log_lock = threading.Lock()
        sync_barrier = threading.Barrier(self.num_of_actor_threads + self.num_of_learner_threads)

        global_agents = []
        for k in range(self.num_of_learner_threads):
            global_agents.append(self.agent)

        actor_threads = []
        for k in range(self.num_of_actor_threads):
            t = actor_thread(
                    thread_num=k,
                    device=self.actor_device,
                    global_memory=self.memory,
                    environment=self.environment[k],
                    global_agents=global_agents,
                    transforms=self.transforms,
                    tensor_board=self.tensor_board,
                    memory_lock=memory_lock,
                    agent_lock=agent_lock,
                    log_lock=log_lock,
                    sync_barrier=sync_barrier,
                    steps=self.steps,
                    epochs=self.epochs,
                    action_random_prob=self.action_random_prob[k])
            actor_threads.append(t)

        learner_threads = []
        for k in range(self.num_of_learner_threads):
            t = learner_thread(
                    thread_num=k,
                    device=self.learner_device,
                    global_memory=self.memory,
                    global_agents=global_agents,
                    tensor_board=self.tensor_board,
                    memory_lock=memory_lock,
                    agent_lock=agent_lock,
                    log_lock=log_lock,
                    sync_barrier=sync_barrier,
                    steps=self.steps,
                    epochs=self.epochs,
                    batch_size=self.batch_size)
            learner_threads.append(t)

        for t in actor_threads:
            t.start()

        for t in learner_threads:
            t.start()

        for t in actor_threads:
            t.join()

        for t in learner_threads:
            t.join()

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