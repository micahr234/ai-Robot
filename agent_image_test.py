from pathlib import Path
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import copy
from memory import *
import numpy as np  # for debug
import matplotlib.pyplot as plt


# Define agent
class agent():

    def __init__(
            self,
            name,
            latent_states,

            latent_net,

            tensor_board,

            latent_learn_rate,

            batches,
            batch_size,
            memory_buffer_size,
            log_level,
            gpu
    ):

        self.name = str(name)
        self.latent_states = latent_states
        self.latent_net_structure = latent_net
        self.tensor_board = tensor_board
        self.batches = batches
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.log_level = log_level
        self.latent_learn_rate = latent_learn_rate

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.latent_filename = self.memory_dir / 'latent.pt'

        self.train_device = torch.device('cpu')
        if torch.cuda.is_available():
            if gpu is None:
                num_of_gpus = torch.cuda.device_count()
                gpu = int(torch.randint(low=0, high=num_of_gpus, size=[]))
            self.train_device = torch.device('cuda:' + str(gpu))
        self.act_device = torch.device('cpu')
        print('Train device:', self.train_device)
        print('Act device:', self.act_device)

        self.build_latent_network()

        self.learn_count = 1
        self.action_count = 1

    def act(self, observation_direct, observation_indirect):

        with torch.no_grad():
            action = torch.rand([1, 2]) * 2 - 1

        self.action_count += 1

        return action

    @staticmethod
    def sample_data(data, data_length, batch_size):

        index = torch.randint(0, data_length, (batch_size,))
        kwargs = {}
        for key, value in data.items():
            kwargs[key] = value[index, :]

        return kwargs

    def learn(self, data):

        data_length = data['survive'].shape[0]
        print('Learning fom ' + str(data_length) + ' samples')

        for batch_num in range(1, self.batches + 1):

            # get data for batch
            with torch.no_grad():
                batch_data = self.sample_data(data, data_length, self.batch_size)
                observation_direct = batch_data['observation_direct'].to(self.train_device, non_blocking=True)
                observation_indirect = batch_data['observation_indirect'].to(self.train_device, non_blocking=True)
                observation_indirect_target = batch_data['observation_indirect_target'].to(self.train_device, non_blocking=True)
                action = batch_data['action'].to(self.train_device, non_blocking=True)
                observation_next_direct = batch_data['observation_next_direct'].to(self.train_device, non_blocking=True)
                observation_next_indirect = batch_data['observation_next_indirect'].to(self.train_device, non_blocking=True)
                observation_next_indirect_target = batch_data['observation_next_indirect_target'].to(self.train_device, non_blocking=True)
                reward = batch_data['reward'].to(self.train_device, non_blocking=True)
                survive = batch_data['survive'].to(self.train_device, non_blocking=True)

            if not self.no_latent_net:
                self.learn_latent(observation_direct, observation_indirect, observation_indirect_target, action,
                                  observation_next_direct, observation_next_indirect, observation_next_indirect_target,
                                  reward, survive)

            self.learn_count += 1

        # print summary
        print('Agent finished learning')

    def learn_latent(self, observation_direct, observation_indirect, observation_indirect_target, action,
                     observation_next_direct, observation_next_indirect, observation_next_indirect_target,
                     reward, survive):

        self.latent_net.train()

        # latent forward pass
        state_latent = self.latent_net.forward_multi(observation_direct, observation_indirect)
        observation_indirect_lastframe = observation_indirect[:, [0], :]
        state_latent_lastframe = state_latent[:, [0], :]

        observation_indirect_target_lastframe = observation_indirect_target[:, [0], :]
        #state_latent_lastframe_lim = state_latent_lastframe[:, :, -observation_indirect_target_lastframe.shape[2]:]

        # optimize latent
        self.latent_optimizer.zero_grad()
        #target_loss = torch.nn.functional.binary_cross_entropy_with_logits(state_latent_lastframe, observation_indirect_target_lastframe.detach())
        target_loss = torch.nn.functional.mse_loss(state_latent_lastframe, observation_indirect_target_lastframe.detach())
        latent_loss = target_loss
        latent_loss.backward()
        self.latent_optimizer.step()

        # update latent learn rates
        self.latent_scheduler.step()

        # log latent results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_latent/target_loss', target_loss.item(), self.learn_count)
            p = self.latent_net(observation_direct[[0], 0, :], observation_indirect_lastframe[[0], 0, :])
            img = list(self.latent_net.children())[0].debug_img
            img_grid = torchvision.utils.make_grid([img[0, [0], :, :], observation_indirect_lastframe[0, 0, [0], :, :]])
            self.tensor_board.add_image('block', img_grid, global_step=self.learn_count)
            img_grid = torchvision.utils.make_grid([img[0, [1], :, :], observation_indirect_lastframe[0, 0, [0], :, :]])
            self.tensor_board.add_image('ball', img_grid, global_step=self.learn_count)
            #img_grid = torchvision.utils.make_grid([observation_indirect_target_lastframe[0, 0, [0], :, :], torch.sigmoid(state_latent_lastframe[0, 0, [0], :, :]), observation_indirect_lastframe[0, 0, [0], :, :]])
            #self.tensor_board.add_image('block', img_grid, global_step=self.learn_count)
            #img_grid = torchvision.utils.make_grid([observation_indirect_target_lastframe[0, 0, [1], :, :], torch.sigmoid(state_latent_lastframe[0, 0, [1], :, :]), observation_indirect_lastframe[0, 0, [0], :, :]])
            #self.tensor_board.add_image('ball', img_grid,  global_step=self.learn_count)

        # log latent learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_latent/learn_rate', self.latent_scheduler.get_last_lr()[0], self.learn_count)

    def save(self):

        print('Saving network and experience')
        torch.save(self.latent_net.state_dict(), self.latent_filename)

    def build_latent_network(self):

        class Latent_Net(torch.nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, observation_direct, observation_indirect):

                state_latent_partial = self.net(observation_indirect)
                #state_latent = torch.cat((observation_direct, state_latent_partial), dim=-1)
                state_latent = state_latent_partial

                return state_latent

            def forward_multi(self, multi_observation_direct, multi_observation_indirect):

                multi_count = multi_observation_indirect.shape[1]
                assert multi_count == multi_observation_direct.shape[1]

                multi_observation_indirect_shape = list(multi_observation_indirect.shape)
                del multi_observation_indirect_shape[1]
                multi_observation_indirect_shape[0] = multi_count * multi_observation_indirect_shape[0]
                multi_observation_indirect = multi_observation_indirect.view(multi_observation_indirect_shape)

                multi_observation_direct_shape = list(multi_observation_direct.shape)
                del multi_observation_direct_shape[1]
                multi_observation_direct_shape[0] = multi_count * multi_observation_direct_shape[0]
                multi_observation_direct = multi_observation_direct.view(multi_observation_direct_shape)

                multi_state_latent = self.forward(multi_observation_direct, multi_observation_indirect)

                multi_state_latent_shape = list(multi_state_latent.shape)
                multi_state_latent_shape[0] = -1
                multi_state_latent_shape.insert(1, multi_count)
                multi_state_latent = multi_state_latent.view(multi_state_latent_shape)

                return multi_state_latent

        self.latent_net = Latent_Net(self.latent_net_structure).to(self.train_device)

        if self.latent_filename.is_file():
            # Load latent network
            print('Loading latent network from file ' + str(self.latent_filename))
            self.latent_net.load_state_dict(torch.load(self.latent_filename))

        else:
            # Build latent network
            print('No latent network loaded from file')

        self.no_latent_net = True if sum(p.numel() for p in self.latent_net.parameters()) == 0 else False

        if not self.no_latent_net:
            self.latent_optimizer = torch.optim.Adam(self.latent_net.parameters(), lr=1.0)
            self.latent_scheduler = torch.optim.lr_scheduler.LambdaLR(self.latent_optimizer, self.latent_learn_rate, last_epoch=-1)