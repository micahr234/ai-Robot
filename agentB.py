import numpy
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from memory import *

# Define agent
class AgentB():

    # ------------------------- Initialization -------------------------

    def __init__(
                self,
                name,
                latent_states,

                latent_fwd_net,
                model_net,
                reward_net,
                survive_net,
                value_net,
                policy_net,

                tensor_board,
                state_input_transform,
                reward_input_transform,
                action_input_transform,
                survive_input_transform,
                action_output_transform,

                latent_learn_rate,
                model_learn_rate,
                reward_learn_rate,
                survive_learn_rate,
                value_learn_rate,
                value_next_learn_factor,
                policy_learn_rate,
                policy_learn_noise_std,

                batches,
                batch_size,
                memory_buffer_size,
                log_level,
                ):

        self.name = str(name)
        self.latent_states = latent_states
        self.latent_fwd_net_structure = latent_fwd_net
        self.value_net_structure = value_net
        self.policy_net_structure = policy_net
        self.model_net_structure = model_net
        self.reward_net_structure = reward_net
        self.survive_net_structure = survive_net
        self.tensor_board = tensor_board
        self.state_input_transform = state_input_transform
        self.reward_input_transform = reward_input_transform
        self.action_input_transform = action_input_transform
        self.survive_input_transform = survive_input_transform
        self.action_output_transform = action_output_transform
        self.batches = batches
        self.memory_buffer_size = memory_buffer_size
        self.batch_size = batch_size
        self.log_level = log_level
        self.latent_learn_rate = latent_learn_rate
        self.policy_learn_rate = policy_learn_rate
        self.policy_learn_entropy_factor = policy_learn_noise_std
        self.value_learn_rate = value_learn_rate
        self.value_next_learn_factor = value_next_learn_factor
        self.model_learn_rate = model_learn_rate
        self.reward_learn_rate = reward_learn_rate
        self.survive_learn_rate = survive_learn_rate

        self.memory_dir = Path.cwd() / 'memory' / self.name
        Path(self.memory_dir).mkdir(parents=True, exist_ok=True)

        self.latent_filename = self.memory_dir / 'latent.pt'
        self.value_filename = self.memory_dir / 'value.pt'
        self.policy_filename = self.memory_dir / 'policy.pt'
        self.env_filename = self.memory_dir / 'env.pt'
        self.memory_buffer_filename = self.memory_dir / 'memory.pt'

        self.train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.act_device = torch.device('cpu')
        print('Train device:', self.train_device)
        print('Act device:', self.act_device)

        self.build_latent_network()
        self.build_env_network()
        self.build_policy_network()
        self.build_value_network()
        self.env_net_slow_copy = copy.deepcopy(self.env_net).to(self.train_device)
        self.latent_net_action_copy = copy.deepcopy(self.latent_net).to(self.act_device)
        self.policy_net_action_copy = copy.deepcopy(self.policy_net).to(self.act_device)

        self.memory = Memory(self.memory_buffer_size, self.memory_buffer_filename)

        self.batch_count = 1
        self.record_count = 1
        self.cumulative_reward = 0

        pass

    # ------------------------- Externally Callable Functions -------------------------

    def act(self, in_state):

        state = self.state_input_transform(in_state).detach().to(self.act_device)

        self.latent_net_action_copy.eval()
        self.policy_net_action_copy.eval()

        with torch.no_grad():
            state_latent = self.latent_net_action_copy(state)
            action = self.policy_net_action_copy(state_latent).sample()

        out_action = self.action_output_transform(action[0, :].cpu())

        return out_action

    def record(self, in_state, in_action, in_reward, in_state_next, in_survive):

        with torch.no_grad():
            state = self.state_input_transform(in_state).cpu()
            action = self.action_input_transform(in_action).cpu()
            reward = self.reward_input_transform(in_reward).cpu()
            state_next = self.state_input_transform(in_state_next).cpu()
            survive = self.survive_input_transform(in_survive).cpu()
            self.memory.add(state, action, reward, state_next, survive)

        self.record_count += 1

        pass

    def learn(self):

        print('Agent ' + str(self.name) + ' learning fom ' + str(len(self.memory)) + ' samples')

        self.latent_net.train()
        self.value_net.train()
        self.policy_net.train()
        self.env_net.train()


        for batch_num in range(1, self.batches + 1):

            # get data for batch
            state, action, reward, state_next, survive = self.memory.sample(self.batch_size)
            state = state.to(self.train_device, non_blocking=True)
            action = action.to(self.train_device, non_blocking=True)
            reward = reward.to(self.train_device, non_blocking=True)
            state_next = state_next.to(self.train_device, non_blocking=True)
            survive = survive.to(self.train_device, non_blocking=True)

            self.learn_latent(state, action, state_next, reward, survive)

            with torch.no_grad():
                state_latent = self.latent_net(state)
                state_next_latent = self.latent_net(state_next)

            self.learn_env(state_latent, action, state_next_latent, reward, survive)

            self.learn_value(state_latent, action, state_next_latent, reward, survive)
            self.learn_policy(state_latent)

            self.batch_count += 1

        self.policy_net_action_copy.load_state_dict(self.policy_net.state_dict())
        self.latent_net_action_copy.load_state_dict(self.latent_net.state_dict())

        # print summary
        print('Agent finished learning')

        pass

    @staticmethod
    def polyak_averaging(net_delayed, net, polyak):

        net_delayed_dict = net_delayed.state_dict()
        net_dict = net.state_dict()

        for key in net_dict:
            net_delayed_dict[key].mul_(polyak)
            net_delayed_dict[key].add_((1 - polyak) * net_dict[key])

        net_delayed.load_state_dict(net_delayed_dict)

        pass

    def learn_latent(self, state, action, state_next, reward, survive):

        # latent forward pass
        state_latent = self.latent_net(state)
        state_next_latent = self.latent_net(state_next)

        state_next_latent_last_frame_prediction_dist, reward_prediction_dist, survive_prediction_dist = self.env_net(state_latent, action)
        state_next_latent_last_frame = self.env_net.model_extract_last_frame(state_next_latent)

        # optimize latent
        self.latent_optimizer.zero_grad()
        state_next_predictive_loss = -torch.mean(state_next_latent_last_frame_prediction_dist.log_prob(state_next_latent_last_frame))
        reward_predictive_loss = -torch.mean(reward_prediction_dist.log_prob(reward))
        survive_predictive_loss = -torch.mean(survive_prediction_dist.log_prob(survive))
        latent_loss = state_next_predictive_loss + reward_predictive_loss + survive_predictive_loss
        latent_loss.backward()
        self.latent_optimizer.step()

        # update latent learn rates
        self.latent_scheduler.step()

        # log latent results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_latent/state_next_predictive_log_loss', state_next_predictive_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_latent/reward_predictive_log_loss', reward_predictive_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_latent/survive_predictive_log_loss', survive_predictive_loss.item(), self.batch_count)

        # log latent learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_latent/learn_rate', self.latent_scheduler.get_last_lr()[0], self.batch_count)

        pass

    def learn_env(self, state_latent, action, state_next_latent, reward, survive):

        # env forward pass
        state_next_latent_last_frame_prediction_dist, reward_prediction_dist, survive_prediction_dist = self.env_net(state_latent, action)
        state_next_latent_last_frame = self.env_net.model_extract_last_frame(state_next_latent)
        #with torch.no_grad():
        #    state_next_latent_prediction_slow, reward_prediction_slow, survive_prediction_slow_dist = self.env_net_slow_copy(state_latent, action)

        # Calculate intrinsic reward
        with torch.no_grad():
            #model_novelty = torch.mean(norm(abs(state_next_latent_prediction_slow[:, -1, :] - state_next_latent_prediction[:, -1, :]), dim=0), dim=-1, keepdim=True)
            #survive_novelty = norm(abs(survive_prediction_slow - survive_prediction), dim=0)
            #reward_novelty = norm(abs(reward_prediction_slow - reward_prediction), dim=0)
            #reward_norm = norm(reward, dim=0)
            #reward_total = reward_norm + survive * 0.01 + survive_novelty * 0.01 + model_novelty * 0.01 + reward_novelty * 0.01
            #reward_total = reward - (1.0 - survive) * 10
            reward_total = reward

        # optimize env
        self.env_optimizer.zero_grad()
        model_loss = -torch.mean(state_next_latent_last_frame_prediction_dist.log_prob(state_next_latent_last_frame))
        model_loss.backward()
        reward_loss = -torch.mean(reward_prediction_dist.log_prob(reward))
        reward_loss.backward()
        survive_loss = -torch.mean(survive_prediction_dist.log_prob(survive))
        survive_loss.backward()
        self.env_optimizer.step()
        #self.polyak_averaging(self.env_net_slow_copy, self.env_net, 0.999)

        # update model learn rates
        self.env_scheduler.step()

        # log env results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_env/model_log_loss', model_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_env/reward_log_loss', reward_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_env/survive_log_loss', survive_loss.item(), self.batch_count)

        # log model learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_env/model_learn_rate', self.env_scheduler.get_last_lr()[0], self.batch_count)
            self.tensor_board.add_scalar('learn_env/reward_learn_rate', self.env_scheduler.get_last_lr()[1], self.batch_count)
            self.tensor_board.add_scalar('learn_env/survive_learn_rate', self.env_scheduler.get_last_lr()[2], self.batch_count)

        pass

    def learn_value(self, state_latent, action, state_next_latent, reward, survive):

        # value forward pass
        model_free_steps = 1
        model_steps = 0
        steps = model_free_steps + model_steps
        state_latent_hypo = [None] * steps
        action_hypo = [None] * steps
        state_next_latent_hypo = [None] * steps
        action_next_hypo = [None] * steps
        value_hypo = [None] * steps
        value_next_hypo = [None] * steps
        value_diff_hypo = [None] * steps
        reward_hypo = [None] * steps
        survive_hypo = [None] * steps
        value_loss = [None] * steps

        for i in range(steps):

            if i < model_free_steps:
                with torch.no_grad():
                    state_latent_hypo[i] = state_latent
                    action_hypo[i] = action
                    state_next_latent_hypo[i], reward_hypo[i], survive_hypo[i] = (state_next_latent, reward, survive)
                    action_next_hypo[i] = self.policy_net(state_next_latent_hypo[i]).sample()
            else:
                with torch.no_grad():
                    first_loop = True if i == 0 else False
                    state_latent_hypo[i] = state_latent if first_loop else state_next_latent_hypo[i-1]
                    action_hypo[i] = self.policy_net(state_latent_hypo[i]).sample() if first_loop else action_next_hypo[i-1]
                    state_next_latent_hypo[i], reward_hypo[i], survive_hypo[i] = self.env_net(state_latent_hypo[i], action_hypo[i])#need to fix
                    action_next_hypo[i] = self.policy_net(state_next_latent_hypo[i]).sample()

            value_hypo[i] = self.value_net(state_latent_hypo[i], action_hypo[i])
            value_next_hypo[i] = self.value_net(state_next_latent_hypo[i], action_next_hypo[i])
            value_diff_hypo[i] = value_hypo[i] - value_next_hypo[i] * survive_hypo[i]

            # optimize value
            self.value_optimizer.zero_grad()
            value_next_hypo[i].register_hook(lambda grad: grad * self.value_scheduler_next_learn_factor)
            value_loss[i] = torch.nn.functional.mse_loss(value_diff_hypo[i], reward_hypo[i])
            value_loss[i].backward()
            self.value_optimizer.step()

        # update value learn rates
        self.value_scheduler.step()
        self.value_scheduler_next_learn_factor = self.value_next_learn_factor(self.value_scheduler.last_epoch)

        # log value results
        if self.log_level >= 1:
            with torch.no_grad():
                value_avg = torch.mean(torch.cat(value_hypo, dim=0))
                value_loss_avg = torch.mean(torch.tensor(value_loss))
            self.tensor_board.add_scalar('learn_value/avg', value_avg.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_value/loss', value_loss_avg.item(), self.batch_count)

        # log value learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_value/learn_rate', self.value_scheduler.get_last_lr()[0], self.batch_count)
            self.tensor_board.add_scalar('learn_value/next_learn_factor', self.value_scheduler_next_learn_factor, self.batch_count)

        pass

    def learn_policy(self, state_latent):

        # policy forward pass
        action_dist = self.policy_net(state_latent)
        action = action_dist.rsample()
        value = self.value_net(state_latent, action)

        # optimize policy
        self.policy_optimizer.zero_grad()
        policy_value_loss = -torch.mean(value)
        policy_entropy_loss = -torch.mean(action_dist.base_dist.entropy())
        policy_entropy_loss.register_hook(lambda grad: grad * 0.0001)
        policy_loss = policy_value_loss + policy_entropy_loss
        policy_loss.backward()
        self.policy_optimizer.step()

        # update policy learn rates
        self.policy_scheduler.step()
        self.policy_scheduler_entropy_factor = self.policy_learn_entropy_factor(self.policy_scheduler.last_epoch)

        # log policy results
        if self.log_level >= 1:
            self.tensor_board.add_scalar('learn_policy/value_loss', policy_value_loss.item(), self.batch_count)
            self.tensor_board.add_scalar('learn_policy/entropy_loss', policy_entropy_loss.item(), self.batch_count)

        # log policy learn rates
        if self.log_level >= 2:
            self.tensor_board.add_scalar('learn_policy/learn_rate', self.policy_scheduler.get_last_lr()[0], self.batch_count)
            self.tensor_board.add_scalar('learn_policy/entropy_factor', self.policy_scheduler_entropy_factor, self.batch_count)

        pass

    def save(self):

        print('Saving network and experience')
        torch.save(self.latent_net.state_dict(), self.latent_filename)
        torch.save(self.value_net.state_dict(), self.value_filename)
        torch.save(self.policy_net.state_dict(), self.policy_filename)
        torch.save(self.env_net.state_dict(), self.env_filename)
        self.memory.save()

        pass

    # ------------------------- Network Functions -------------------------

    def build_latent_network(self):

        class Latent_Net(torch.nn.Module):

            def __init__(self, fwd_net):
                super().__init__()
                self.fwd_net = fwd_net
                pass

            def forward(self, state):
                steps = state.shape[1]
                state_latent_list = [None] * steps
                for i in range(state.shape[1]):
                    y = self.fwd_net(state[:, i, :])
                    state_latent_list[i] = y
                    state_latent_list[i] = torch.unsqueeze(state_latent_list[i], 1)
                state_latent = torch.cat(state_latent_list, dim=1)
                return state_latent

        self.latent_net = Latent_Net(self.latent_fwd_net_structure).to(self.train_device)

        if self.latent_filename.is_file():
            # Load latent network
            print('Loading latent network from file ' + str(self.latent_filename))
            self.latent_net.load_state_dict(torch.load(self.latent_filename))

        else:
            # Build latent network
            print('No latent network loaded from file')

        self.latent_optimizer = torch.optim.Adam(self.latent_net.parameters(), lr=1.0)
        self.latent_scheduler = torch.optim.lr_scheduler.LambdaLR(self.latent_optimizer, self.latent_learn_rate, last_epoch=-1)

        pass

    def build_policy_network(self):

        class Policy_Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input):
                y = self.net(state_input.flatten(start_dim=1))
                halfsize = int(y.shape[-1] / 2)
                action_mu = torch.tanh(y[:, :halfsize])
                action_logvar = y[:, halfsize:]
                action_sigma = torch.exp(action_logvar / 2)
                action_dist_pre = torch.distributions.normal.Normal(action_mu, action_sigma)
                transform = torch.distributions.transforms.TanhTransform(cache_size=0)
                action_dist = torch.distributions.transformed_distribution.TransformedDistribution(action_dist_pre, transform)
                return action_dist

        self.policy_net = Policy_Net(self.policy_net_structure).to(self.train_device)

        if self.policy_filename.is_file():
            # Load policy network
            print('Loading policy network from file ' + str(self.policy_filename))
            self.policy_net.load_state_dict(torch.load(self.policy_filename))

        else:
            # Build policy network
            print('No policy network loaded from file')

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1.0)
        self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(self.policy_optimizer, self.policy_learn_rate, last_epoch=-1)
        self.policy_scheduler_entropy_factor = self.policy_learn_entropy_factor(self.policy_scheduler.last_epoch)

        pass

    def build_value_network(self):

        class Value_Net(torch.nn.Module):

            def __init__(self, net):
                super().__init__()
                self.net = net
                pass

            def forward(self, state_input, action_input):
                c = torch.cat((state_input.flatten(start_dim=1), action_input), dim=-1)
                y = self.net(c)
                return y

        self.value_net = Value_Net(self.value_net_structure).to(self.train_device)

        if self.value_filename.is_file():
            # Load value network
            print('Loading value network from file ' + str(self.value_filename))
            self.value_net.load_state_dict(torch.load(self.value_filename))

        else:
            # Build value network
            print('No value network loaded from file')

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1.0)
        self.value_scheduler = torch.optim.lr_scheduler.LambdaLR(self.value_optimizer, self.value_learn_rate, last_epoch=-1)
        self.value_scheduler_next_learn_factor = self.value_next_learn_factor(self.value_scheduler.last_epoch)

        pass

    def build_env_network(self):

        class Env_Net(torch.nn.Module):

            def __init__(self, model_net, reward_net, survive_net):
                super().__init__()
                self.model_net = model_net
                self.reward_net = reward_net
                self.survive_net = survive_net
                pass

            def model_forward(self, state, action):
                c = torch.cat((state.flatten(start_dim=1), action), dim=-1)
                y = self.model_net(c)
                halfsize = int(y.shape[-1] / 2)
                state_next_last_frame_diff_mu = y[:, :halfsize].unsqueeze(1)
                state_next_last_frame_diff_logvar = y[:, halfsize:].unsqueeze(1)
                state_next_last_frame_mu = state_next_last_frame_diff_mu # + state[:, [-1], :]
                state_next_last_frame_logvar = state_next_last_frame_diff_logvar
                state_next_last_frame_sigma = torch.clamp(torch.exp(state_next_last_frame_logvar / 2), min=0.01)
                state_next_last_frame_dist = torch.distributions.normal.Normal(state_next_last_frame_mu, state_next_last_frame_sigma)
                return state_next_last_frame_dist

            def model_cat_next_frame(self, state, state_next_last_frame):
                state_next = torch.cat((state[:, 1:, :], state_next_last_frame), dim=1)
                return state_next

            def model_extract_last_frame(self, state):
                state_last_frame = state[:, [-1], :]
                return state_last_frame

            def reward_forward(self, state, state_next_last_frame_mu, state_next_last_frame_sigma, action):
                c = torch.cat((state.flatten(start_dim=1), state_next_last_frame_mu.flatten(start_dim=1), state_next_last_frame_sigma.flatten(start_dim=1), action), dim=-1)
                y = self.reward_net(c)
                halfsize = int(y.shape[-1] / 2)
                reward_mu = y[:, :halfsize]
                reward_logvar = y[:, halfsize:]
                reward_sigma = torch.clamp(torch.exp(reward_logvar / 2), min=0.01)
                reward_dist = torch.distributions.normal.Normal(reward_mu, reward_sigma)
                return reward_dist

            def survive_forward(self, state, state_next_last_frame, state_next_last_frame_sigma, action):
                c = torch.cat((state.flatten(start_dim=1), state_next_last_frame.flatten(start_dim=1), state_next_last_frame_sigma.flatten(start_dim=1), action), dim=-1)
                y = self.survive_net(c)
                survive_dist = torch.distributions.bernoulli.Bernoulli(logits=y)
                return survive_dist

            def forward(self, state, action):
                state_next_last_frame_dist = self.model_forward(state, action)
                state_next_last_frame_mu_detach = state_next_last_frame_dist.loc.detach()
                state_next_last_frame_logvar_detach = torch.log(state_next_last_frame_dist.variance).detach()
                reward_dist = self.reward_forward(state, state_next_last_frame_mu_detach, state_next_last_frame_logvar_detach, action)
                survive_dist = self.survive_forward(state, state_next_last_frame_mu_detach, state_next_last_frame_logvar_detach, action)
                return state_next_last_frame_dist, reward_dist, survive_dist

        self.env_net = Env_Net(self.model_net_structure, self.reward_net_structure, self.survive_net_structure).to(self.train_device)

        if self.env_filename.is_file():
            # Load environment network
            print('Loading environment network from file ' + str(self.env_filename))
            self.env_net.load_state_dict(torch.load(self.env_filename))

        else:
            # Build environment network
            print('No environment network loaded from file')

        self.env_optimizer = torch.optim.Adam([{'params': self.env_net.model_net.parameters()},
                                               {'params': self.env_net.reward_net.parameters()},
                                               {'params': self.env_net.survive_net.parameters()}
                                               ], lr=1.0)
        self.env_scheduler = torch.optim.lr_scheduler.LambdaLR(self.env_optimizer, (self.model_learn_rate, self.reward_learn_rate, self.survive_learn_rate), last_epoch=-1)

        pass
