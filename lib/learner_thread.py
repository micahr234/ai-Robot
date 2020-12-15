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


class learner_thread(threading.Thread):

    def __init__(self,
            thread_num,
            device,
            global_memory,
            global_agents,
            tensor_board,
            memory_lock,
            agent_lock,
            log_lock,
            sync_barrier,
            steps,
            epochs,
            batch_size):

        threading.Thread.__init__(self)

        self.number = thread_num
        self.name = "LearnerThread" + str(self.number)
        self.device = device
        self.global_memory = global_memory
        self.global_agents = global_agents
        self.tensor_board = tensor_board
        self.memory_lock = memory_lock
        self.agent_lock = agent_lock
        self.log_lock = log_lock
        self.sync_barrier = sync_barrier
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size

        self.learnstep = 1
        self.print_interval = 100
        self.start_learn_threshhold = self.batch_size

        self.agent_lock.acquire()
        self.agent = copy.deepcopy(self.global_agents[self.number])
        self.agent_lock.release()
        self.agent.set_device(self.device)

    def run(self):

        self.log_lock.acquire()
        print("Starting " + self.name)
        self.log_lock.release()

        for e in range(self.epochs):

            for t in range(self.steps):
                self.learn()
                self.learnstep += 1

            self.agent_lock.acquire()
            self.global_agents[self.number] = copy.deepcopy(self.agent)
            self.agent_lock.release()

            self.log_lock.acquire()
            print(self.name + " at barrier")
            self.log_lock.release()

            self.sync_barrier.wait()

            self.log_lock.acquire()
            print(self.name + " passed barrier")
            self.log_lock.release()

        self.log_lock.acquire()
        print("Exiting " + self.name)
        self.log_lock.release()

    def learn(self):

        self.memory_lock.acquire()

        data_length = len(self.global_memory)
        if data_length >= self.start_learn_threshhold:

            with torch.no_grad():
                data = self.global_memory.get_all()
                batch_data = self.sample_data(data, data_length, self.batch_size, self.device)

            self.memory_lock.release()

            (value_avg, value_loss, policy_loss) = self.agent.learn(**batch_data)

            self.log_lock.acquire()
            if (self.learnstep % self.print_interval) == 0:
                print('---' + self.name + ' Learning Complete---\t\t\t\tLearning step: ' + str(self.learnstep))
            self.tensor_board.add_scalar('learn_value/avg_' + self.name, value_avg.item(), self.learnstep)
            self.tensor_board.add_scalar('learn_value/loss_' + self.name, value_loss.item(), self.learnstep)
            self.tensor_board.add_scalar('learn_policy/loss_' + self.name, policy_loss.item(), self.learnstep)
            #self.tensor_board.add_scalar('learn_value/learn_rate', self.value_scheduler.get_last_lr()[0], self.learn_count)
            self.log_lock.release()

        else:

            self.memory_lock.release()

    @staticmethod
    def sample_data(data, data_length, batch_size, device):

        index = torch.randint(0, data_length, (batch_size,))
        kwargs = {}
        for key, value in data.items():
            kwargs[key] = value[index, :].clone().to(device, non_blocking=True)

        return kwargs
