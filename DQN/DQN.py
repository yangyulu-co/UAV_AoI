import os

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from Memory import ReplayMemory
from utils import identity, to_tensor_var


class DQN:
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000, max_episodes=2000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 target_tau=0.01, target_update_steps=50,
                 actor_hidden_size=32, actor_output_act=identity, actor_loss="huber",
                 actor_lr=0.001, actor_lr_end=0.0001, optimizer_type="rmsprop",
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()

        self.memory = ReplayMemory(memory_capacity)
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.target_tau = target_tau  # 软更新相关参数
        self.target_update_steps = target_update_steps

        self.actor_hidden_size = actor_hidden_size
        self.actor_output_act = actor_output_act
        self.actor_lr = actor_lr
        self.actor_lr_end = actor_lr_end
        self.optimizer_type = optimizer_type
        self.max_grad_norm = max_grad_norm
        self.actor_loss = actor_loss

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.actor_target = deepcopy(self.actor)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.actor_target.cuda()

        # self.now_step = 0
        # self.step_record = []
        # self.loss_record = []
        self.episode_done = False

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = self.action(state)
        return action

    # choose an action based on state for execution
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        state_action_value_var = self.actor(state_var)
        if self.use_cuda:
            state_action_value = state_action_value_var.data.cpu().numpy()[0]
        else:
            state_action_value = state_action_value_var.data.numpy()[0]
        action = np.argmax(state_action_value)
        return action

    # soft update the actor target network
    def soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    # take one step and push the data to memory
    def take_one_step(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        state = self.env_state
        action = self.exploration_action(self.env_state)
        next_state, reward, done, _ = self.env.step(action)
        if done:
            if self.done_penalty is not None:
                reward = self.done_penalty
            next_state = [0] * len(state)
            self.env_state = self.env.reset()
            print("episode finish within %d steps" % self.n_steps)
            self.episode_done = True
            self.n_steps = 0
            self.n_episodes += 1
        else:
            self.episode_done = False
            self.env_state = next_state
            self.n_steps += 1
        self.memory.push(state, action, reward, next_state, done)

    # train on a sample batch and update the network
    def train_one_step(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)  # view函数用来reshape
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        current_q = self.actor(states_var).gather(1, actions_var)
        next_state_action_values = self.actor_target(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

        # self.now_step += 1
        # self.step_record.append(self.now_step)
        # self.loss_record.append(np.mean((current_q.detach().cpu().numpy() - target_q.detach().cpu().numpy())) ** 2)

        # update actor network
        self.actor_optimizer.zero_grad()
        if self.actor_loss == "huber":
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            loss = th.nn.MSELoss()(current_q, target_q)
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update target network
        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            self.soft_update_target(self.actor_target, self.actor)

    def train(self):
        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "DQN_model.pth"
        f = open(directory + '/' + "DQN_reward_data.txt", "w")
        f2 = open(directory + '/' + "DQN_saved_time_data.txt", "w")
        while self.n_episodes < self.max_episodes:
            self.take_one_step()  # 与环境交互（step一次）
            if self.n_episodes >= self.episodes_before_train:
                self.train_one_step()
            #  每训练1步计算一个reward
            if self.episode_done and ((self.n_episodes + 1) % 1 == 0):
                print("Training Episode %d" % (self.n_episodes + 1))
                th.save(self.actor.state_dict(), filename)
                test_reward, test_real_target = self.test_one_episode()
                print(test_reward)
                print(test_real_target)
                f.write(f"{test_reward}\n")
                f2.write(f"{test_real_target}\n")
        f.close()
        f2.close()
        # self.writer.flush()

    def test_one_episode(self):
        filename = "Pretrained_model/DQN_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)
        ep_real_target = 0
        ep_reward = 0
        avg_real_target = 0
        avg_reward = 0
        avg_num = 1
        state = self.env.reset()
        for ep in range(avg_num):
            for t in range(1, self.max_steps):
                action = self.action(state)  # 选择动作
                state, reward, done, real_target = self.env.step(action)
                ep_reward += reward
                ep_real_target += real_target
                if done:
                    break
            avg_reward += ep_reward
            avg_real_target += ep_real_target
            ep_reward = 0
            ep_real_target = 0
            state = self.env.reset()
        avg_reward /= avg_num
        avg_real_target /= avg_num
        return avg_reward, avg_real_target

    def test(self):
        filename = "Pretrained_model/DQN_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)

        test_running_reward = 0
        for ep in range(1, 2):
            ep_reward = 0
            ep_reward_record = []
            state = self.env.reset()
            for t in range(1, self.max_steps):
                action = self.action(state)  # 选择动作
                print(action)
                state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                ep_reward_record.append(ep_reward)
                if done:
                    break
            plt.figure()
            plt.plot(ep_reward_record)
            plt.show()
            self.env.render()  # 在屏幕上显示
            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out
