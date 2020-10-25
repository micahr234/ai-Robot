from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
from memory import *
import numpy as np  # for debug


# Define agent
class agent():

    def __init__(
            self,
            name,
            latent_states,
            action_distributions,

            latent_net,
            model_net,
            reward_net,
            survive_net,
            value_net,
            policy_net,
            policy_mix_net,

            tensor_board,

            latent_learn_rate,
            latent_polyak,
            model_learn_rate,
            reward_learn_rate,
            survive_learn_rate,
            env_polyak,
            value_learn_rate,
            value_hallu_loops,
            value_polyak,
            policy_learn_rate,
            policy_polyak,

            batches,
            batch_size,
            memory_buffer_size,
            log_level,
            gpu
    ):

        self.name = str(name)
        self.latent_states = latent_states
        self.action_distributions = action_distributions
        self.latent_net_structure = latent_net
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.policy_mix_net_structure = policy_mix_net
        self.model_net_structure = model_net
        self.reward_net_structure = reward_net
        self.survive_net_structure = survive_net
        self.tensor_board = tensor_board
        self.batches = batches
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.log_level = log_level
        self.latent_learn_rate = latent_learn_rate
        self.latent_polyak = latent_polyak
        self.policy_learn_rate = policy_learn_rate
        self.value_learn_rate = value_learn_rate
        self.value_hallu_loops = value_hallu_loops
        self.model_learn_rate = model_learn_rate
        self.reward_learn_rate = reward_learn_rate
        self.survive_learn_rate = survive_learn_rate
        self.env_polyak = env_polyak
        self.policy_polyak = policy_polyak
        self.value_polyak = value_polyak

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.latent_filename = self.memory_dir / 'latent.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        #policy mix net
        self.env_filename = self.memory_dir / 'env.pt'

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
        self.build_env_network()
        self.build_policy_network()
        self.build_value_network()

        self.latent_net_action_copy = copy.deepcopy(self.latent_net).to(self.act_device)
        self.policy_net_action_copy = copy.deepcopy(self.policy_net).to(self.act_device)

        self.learn_count = 1
        self.action_count = 1

    @staticmethod
    def sample_data(data, data_length, batch_size):

        index = torch.randint(0, data_length, (batch_size,))
        kwargs = {}
        for key, value in data.items():
            kwargs[key] = value[index, :]

        return kwargs

    @staticmethod
    def polyak_averaging(net, target_net, polyak):

        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(polyak * param.data + (1.0 - polyak) * target_param.data)

    def act(self, observation_direct, observation_indirect):

        with torch.no_grad():

            self.latent_net_action_copy.eval()
            self.policy_net_action_copy.eval()

            observation_direct.to(self.act_device)
            observation_indirect.to(self.act_device)

            state_latent = self.latent_net_action_copy.forward_multi(observation_direct, observation_indirect)
            action = self.policy_net_action_copy(state_latent)

        self.action_count += 1

        return action

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

            with torch.no_grad():
                state_latent = self.latent_net.forward_multi(observation_direct, observation_indirect)
                state_next_latent = self.latent_net.forward_multi(observation_next_direct, observation_next_indirect)

            if not self.no_env_net:
                self.learn_env(state_latent, action, state_next_latent, reward, survive)

            self.learn_value(state_latent, action, state_next_latent, reward, survive)
            self.learn_policy(state_latent)

            self.learn_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())
        self.latent_net_action_copy.load_state_dict(self.latent_net.state_dict())

        # print summary
        print('Agent finished learning')

    def learn_latent(self, observation_direct, observation_indirect, observation_indirect_target, action,
                     observation_next_direct, observation_next_indirect, observation_next_indirect_target,
                     reward, survive):

        self.latent_net.train()
        self.latent_target_net.eval()
        self.env_target_net.eval()

        # latent forward pass
        state_latent = self.latent_net.forward_multi(observation_direct, observation_indirect)
        state_latent_lastframe = self.env_target_net.state_extract_lastframe(state_latent)
        state_next_latent_lastframe_prediction, reward_prediction, survive_prediction = self.env_target_net(state_latent, action)

        with torch.no_grad():

            state_next_latent = self.latent_target_net.forward_multi(observation_next_direct, observation_next_indirect)
            state_next_latent_lastframe = self.env_target_net.state_extract_lastframe(state_next_latent)

            observation_indirect_target_lastframe = self.env_target_net.state_extract_lastframe(observation_indirect_target)

        state_latent_lastframe_lim = state_latent_lastframe[:, :, -observation_indirect_target_lastframe.shape[2]:]

        # optimize latent
        self.latent_optimizer.zero_grad()
        state_next_loss = torch.nn.functional.mse_loss(state_next_latent_lastframe_prediction, state_next_latent_lastframe)
        reward_loss = torch.nn.functional.mse_loss(reward_prediction, reward)
        survive_loss = torch.nn.functional.binary_cross_entropy_with_logits(survive_prediction, survive)
        target_loss = torch.nn.functional.mse_loss(state_latent_lastframe_lim, observation_indirect_target_lastframe)
        latent_loss = state_next_loss + reward_loss + survive_loss + target_loss
        latent_loss.backward()
        self.latent_optimizer.step()

        # update target models
        self.polyak_averaging(self.latent_net, self.latent_target_net, self.latent_polyak)

        # update latent learn rates
        self.latent_scheduler.step()

        # log latent results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_latent/state_next_predictive_loss', state_next_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_latent/reward_predictive_loss', reward_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_latent/survive_predictive_loss', survive_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_latent/target_loss', target_loss.item(), self.learn_count)

        # log latent learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_latent/learn_rate', self.latent_scheduler.get_last_lr()[0], self.learn_count)

    def learn_env(self, state_latent, action, state_next_latent, reward, survive):

        self.env_net.train()
        self.env_target_net.eval()

        # env forward pass
        state_next_latent_lastframe_prediction, reward_prediction, survive_prediction = self.env_net(state_latent, action)

        with torch.no_grad():
            state_next_latent_lastframe = self.env_net.state_extract_lastframe(state_next_latent)

        # Calculate intrinsic reward
        with torch.no_grad():
            # state_next_latent_prediction_slow, reward_prediction_slow, survive_prediction_slow_dist = self.env_net_slow_copy(state_latent, action)
            # model_novelty = torch.mean(norm(abs(state_next_latent_prediction_slow[:, -1, :] - state_next_latent_prediction[:, -1, :]), dim=0), dim=-1, keepdim=True)
            # survive_novelty = norm(abs(survive_prediction_slow - survive_prediction), dim=0)
            # reward_novelty = norm(abs(reward_prediction_slow - reward_prediction), dim=0)
            # reward_norm = norm(reward, dim=0)
            # reward_total = reward_norm + survive * 0.01 + survive_novelty * 0.01 + model_novelty * 0.01 + reward_novelty * 0.01
            # reward_total = reward - (1.0 - survive) * 10
            reward_total = reward

        # optimize env
        self.env_optimizer.zero_grad()
        state_next_loss = torch.nn.functional.mse_loss(state_next_latent_lastframe_prediction, state_next_latent_lastframe)
        state_next_loss.backward()
        reward_loss = torch.nn.functional.mse_loss(reward_prediction, reward_total)
        reward_loss.backward()
        survive_loss = torch.nn.functional.binary_cross_entropy_with_logits(survive_prediction, survive)
        survive_loss.backward()
        self.env_optimizer.step()

        # update target models
        self.polyak_averaging(self.env_net, self.env_target_net, self.env_polyak)

        # update model learn rates
        self.env_scheduler.step()

        # log env results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_env/state_next_loss', state_next_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_env/reward_loss', reward_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_env/survive_loss', survive_loss.item(), self.learn_count)

        # log model learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_env/model_learn_rate', self.env_scheduler.get_last_lr()[0], self.learn_count)
            self.tensor_board.add_scalar('learn_env/reward_learn_rate', self.env_scheduler.get_last_lr()[1], self.learn_count)
            self.tensor_board.add_scalar('learn_env/survive_learn_rate', self.env_scheduler.get_last_lr()[2], self.learn_count)

    def learn_value(self, state_latent, action, state_next_latent, reward, survive):

        self.env_net.eval()
        self.value_net.train()
        self.value_target_net.eval()
        self.policy_target_net.eval()

        # value forward pass
        value_buffer = []
        value_loss_buffer = []

        for i in range(self.value_hallu_loops):

            with torch.no_grad():

                if i == 0:
                    state_latent_hallu = state_latent
                    action_hallu = action
                    state_next_latent_hallu, reward_hallu, survive_prob_hallu = (state_next_latent, reward, survive)
                    #state_next_latent_lastframe_hallu, reward_hallu, survive_hallu = self.env_net(state_latent_hallu, action_hallu)
                    #state_next_latent_hallu = self.env_net.state_cat_nextframe(state_latent_hallu, state_next_latent_lastframe_hallu)
                    #survive_prob_hallu = torch.sigmoid(survive_hallu)

                else:
                    state_latent_hallu = state_next_latent_hallu.unsqueeze(0).repeat(self.value_action_samples, 1, 1, 1).flatten(start_dim=0, end_dim=1)
                    action_hallu = action_next_hallu.flatten(start_dim=0, end_dim=1)
                    state_next_latent_lastframe_hallu, reward_hallu, survive_hallu = self.env_net(state_latent_hallu, action_hallu)
                    state_next_latent_hallu = self.env_net.state_cat_nextframe(state_latent_hallu, state_next_latent_lastframe_hallu)
                    survive_prob_hallu = torch.sigmoid(survive_hallu)

                action_next_hallu, action_next_probs_hallu, _ = self.policy_target_net.forward_train(state_next_latent_hallu)

            value_hallu = self.value_net(state_latent_hallu, action_hallu)

            with torch.no_grad():
                value_next_target_hallu = torch.sum(self.value_target_net.forward_multi_action(state_next_latent_hallu, action_next_hallu) * action_next_probs_hallu, dim=0)
                value_target_hallu = reward_hallu + value_next_target_hallu * survive_prob_hallu * 0.99

            # optimize value
            self.value_optimizer.zero_grad()
            value_loss = torch.nn.functional.mse_loss(value_hallu, value_target_hallu)
            value_loss.backward()
            self.value_optimizer.step()

            with torch.no_grad():
                value_buffer.append(value_hallu)
                value_loss_buffer.append(value_loss.unsqueeze(0))

        # update target models
        self.polyak_averaging(self.value_net, self.value_target_net, self.value_polyak)

        # update value learn rates
        self.value_scheduler.step()

        # log value results
        if self.log_level >= 1:
            with torch.no_grad():
                value_avg = torch.mean(torch.cat(value_buffer, dim=0))
                value_loss_avg = torch.mean(torch.cat(value_loss_buffer, dim=0))
            self.tensor_board.add_scalar('learn_value/avg', value_avg.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_value/loss', value_loss_avg.item(), self.learn_count)

        # log value learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_value/learn_rate', self.value_scheduler.get_last_lr()[0], self.learn_count)

    def learn_policy(self, state_latent):

        self.value_net.eval()
        self.policy_net.train()

        # policy forward pass
        action, action_mix_prob, action_mix_entropy = self.policy_net.forward_train(state_latent)
        value_each = self.value_net.forward_multi_action(state_latent, action)
        value = torch.sum(value_each * action_mix_prob.detach(), dim=0)

        # optimize policy
        self.policy_optimizer.zero_grad()
        value_loss = -torch.mean(value)
        diversity_loss = -torch.mean(torch.log(torch.var(action, dim=0))) * 0.01
        policy_loss = value_loss + diversity_loss
        policy_loss.backward()
        self.policy_optimizer.step()

        self.policy_mix_optimizer.zero_grad()
        mix_value_loss = -torch.mean(torch.log(action_mix_prob) * value_each.detach())
        mix_diversity_loss = -torch.mean(action_mix_entropy)
        policy_mix_loss = mix_value_loss + mix_diversity_loss
        policy_mix_loss.backward()
        self.policy_mix_optimizer.step()

        # update target models
        self.polyak_averaging(self.policy_net, self.policy_target_net, self.policy_polyak)

        # update policy learn rates
        self.policy_scheduler.step()

        # log policy results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_policy/value_loss', value_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_policy/diversity_loss', diversity_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_policy/mix_loss', mix_value_loss.item(), self.learn_count)
            self.tensor_board.add_scalar('learn_policy/entropy_loss', mix_diversity_loss.item(), self.learn_count)

        # log policy learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_policy/learn_rate', self.policy_scheduler.get_last_lr()[0], self.learn_count)

    def save(self):

        print('Saving network and experience')
        torch.save(self.latent_net.state_dict(), self.latent_filename)
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)
        torch.save(self.env_net.state_dict(), self.env_filename)

    def build_latent_network(self):

        class Latent_Net(torch.nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, observation_direct, observation_indirect):

                state_latent_partial = self.net(observation_indirect)
                state_latent = torch.cat((observation_direct, state_latent_partial), dim=-1)

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

        self.latent_target_net = copy.deepcopy(self.latent_net)

        self.no_latent_net = True if sum(p.numel() for p in self.latent_net.parameters()) == 0 else False

        if not self.no_latent_net:
            self.latent_optimizer = torch.optim.Adam(self.latent_net.parameters(), lr=1.0)
            self.latent_scheduler = torch.optim.lr_scheduler.LambdaLR(self.latent_optimizer, self.latent_learn_rate, last_epoch=-1)

    def build_env_network(self):

        class Env_Net(torch.nn.Module):

            def __init__(self, model_net, reward_net, survive_net):

                super().__init__()
                self.model_net = model_net
                self.reward_net = reward_net
                self.survive_net = survive_net

            def model_forward(self, state, action):

                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                y = self.model_net(c)
                state_next_lastframe = y.unsqueeze(1)

                return state_next_lastframe

            @staticmethod
            def state_cat_nextframe(state, state_next_lastframe):

                state_next = torch.cat((state_next_lastframe, state[:, :-1, :]), dim=1)

                return state_next

            @staticmethod
            def state_extract_lastframe(state):

                state_lastframe = state[:, [0], :]

                return state_lastframe

            def reward_forward(self, state, action):

                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                reward = self.reward_net(c)

                return reward

            def survive_forward(self, state, action):

                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                survive = self.survive_net(c)

                return survive

            def forward(self, state, action):

                state_next_lastframe = self.model_forward(state, action)
                reward = self.reward_forward(state, action)
                survive = self.survive_forward(state, action)

                return state_next_lastframe, reward, survive

        self.env_net = Env_Net(self.model_net_structure, self.reward_net_structure, self.survive_net_structure).to(self.train_device)

        if self.env_filename.is_file():
            # Load environment network
            print('Loading environment network from file ' + str(self.env_filename))
            self.env_net.load_state_dict(torch.load(self.env_filename))

        else:
            # Build environment network
            print('No environment network loaded from file')

        self.env_target_net = copy.deepcopy(self.env_net)

        self.no_env_net = True if sum(p.numel() for p in self.env_net.model_net.parameters()) == 0 else False

        if not self.no_env_net:
            self.env_optimizer = torch.optim.Adam([{'params': self.env_net.model_net.parameters()},
                                                   {'params': self.env_net.reward_net.parameters()},
                                                   {'params': self.env_net.survive_net.parameters()}
                                                   ], lr=1.0)
            self.env_scheduler = torch.optim.lr_scheduler.LambdaLR(self.env_optimizer, (self.model_learn_rate, self.reward_learn_rate, self.survive_learn_rate), last_epoch=-1)

    def build_value_network(self):

        class Value_Net(torch.nn.Module):

            def __init__(self, net):

                super().__init__()
                self.net = net

            def forward(self, state, action):

                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                y = self.net(c)

                return y

            def forward_multi_action(self, state, multi_action):

                multi_count = multi_action.shape[0]
                multi_state = state.unsqueeze(0).repeat(multi_count, 1, 1, 1)

                multi_state_shape = list(multi_state.shape)
                del multi_state_shape[0]
                multi_state_shape[0] = -1
                multi_state = multi_state.view(multi_state_shape)

                multi_action_shape = list(multi_action.shape)
                del multi_action_shape[0]
                multi_action_shape[0] = -1
                multi_action = multi_action.view(multi_action_shape)

                multi_value = self.forward(multi_state, multi_action)

                multi_value_shape = list(multi_value.shape)
                multi_value_shape[0] = -1
                multi_value_shape.insert(0, multi_count)
                multi_value = multi_value.view(multi_value_shape)

                return multi_value

        self.value_net = Value_Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_target_net = copy.deepcopy(self.value_net)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1.0)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, self.value_learn_rate, last_epoch=-1)

    def build_policy_network(self):

        class Policy_Net(torch.nn.Module):

            def __init__(self, net, net_mix, action_distributions):

                super().__init__()
                self.action_distributions = action_distributions
                self.net_mix = net_mix
                self.net = net

            def distribution(self, state):

                state_flat = state.flatten(start_dim=1)

                y = self.net(state_flat).unsqueeze(-1)

                if torch.isnan(y).sum() > 0:
                    raise ValueError('Nan values in actions')

                y_shape = list(y.shape)
                y_shape[-2] = -1
                y_shape[-1] = self.action_distributions
                action = torch.tanh(y.view(y_shape))

                x = self.net_mix(state_flat).unsqueeze(1)

                action_mix = x
                action_mix_dist = torch.distributions.categorical.Categorical(logits=action_mix)

                return action, action_mix_dist

            def forward(self, state, samples=None):

                action_raw, action_mix_dist = self.distribution(state)

                samples_actual = 1 if samples is None else samples
                num_of_actions = action_raw.shape[1]
                action_mix = action_mix_dist.sample([samples_actual]).repeat(1, 1, num_of_actions)
                action_comp = action_raw.unsqueeze(0).repeat(samples_actual, 1, 1, 1)

                action = torch.gather(action_comp, -1, action_mix.unsqueeze(-1)).squeeze(-1)
                action = action.squeeze(0) if samples is None else action

                return action

            def forward_train(self, state):

                action_raw, action_mix_dist = self.distribution(state)

                action_mix_prob = action_mix_dist.probs.permute([2, 0, 1]).contiguous()
                action_mix_entropy = action_mix_dist.entropy()
                action = action_raw.permute([2, 0, 1]).contiguous()

                return action, action_mix_prob, action_mix_entropy

        self.policy_net = Policy_Net(self.policy_net_structure, self.policy_mix_net_structure, self.action_distributions).to(self.train_device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_target_net = copy.deepcopy(self.policy_net)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.net.parameters(), lr=1.0)
        self.policy_mix_optimizer = torch.optim.Adam(self.policy_net.net_mix.parameters(), lr=1.0)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, self.policy_learn_rate, last_epoch=-1)
