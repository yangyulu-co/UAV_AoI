import os

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

from matplotlib import pyplot as plt

from Memory import ReplayMemory
from utils import index_to_one_hot, to_tensor_var


class PPO:
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None, max_episodes=2000,
                 roll_out_n_steps=1, target_tau=1.,
                 clip_param=0.2, target_update_steps=5,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
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
        self.roll_out_n_steps = roll_out_n_steps  # TD-n,交互n步后存入memory
        self.max_episodes = max_episodes

        self.clip_param = clip_param
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    # 得到state下各action的概率
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # 选择一个动作（epsilon-greedy）
    def exploration_action(self, state):
        action_prob = self._softmax_action(state)  # get_action_prob用来取得action概率
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.n_steps / self.epsilon_decay)
        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(action_prob)
        return action

    # 选择一个动作（无探索）
    def action(self, state):
        action_prob = self._softmax_action(state)
        action = np.argmax(action_prob)
        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def _take_n_steps(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(deepcopy(self.env_state))
            action = self.exploration_action(self.env_state)
            next_state, reward, done = self.env.step(action)
            actions.append(action)
            if done and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done:
            final_value = 0.0
            print("episode finish within %d steps" % (self.n_steps))
            self.n_steps = 0
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_value = self.value(final_state, final_action)
        rewards = self._discount_reward(rewards, final_value)
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

    # discount roll out rewards: 当采取n_step时，获得一个Q序列
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # agent interact with the environment to collect experience
    def interact(self):
        # TD-n,交互n步后将每一步存入memory
        self._take_n_steps()

    # train on a sample batch: 执行从memory中提取一个batch数据，并对actor的网络进行一次更新
    def train_one_step(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        Qs_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        values = self.critic_target(states_var, actions_var).detach()  # Critic_target网络计算当前V
        advantages = Qs_var - values  # 优势函数
        action_log_probs = self.actor(states_var)  # actor产生的动作概率(log)，batch_size * action_dim
        action_log_probs = th.sum(action_log_probs * actions_var, 1)  # 每一步以多大概率采取当前措施（batch_size维向量）
        old_action_log_probs = self.actor_target(states_var).detach()  # actor_target产生的动作概率
        old_action_log_probs = th.sum(old_action_log_probs * actions_var, 1)  # actor_target以多大概率采取当前措施
        ratio = th.exp(action_log_probs - old_action_log_probs)  # 重要性采样的系数
        surr1 = ratio * advantages  #
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages  # 给重要性采样的系数限定一个范围
        actor_loss = -th.mean(th.min(surr1, surr2))  # 重要性采样过后优势函数的平均
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = Qs_var
        values = self.critic(states_var, actions_var)
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # update actor target network and critic target network
        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

    def train(self):
        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "PPO_model.pth"
        f = open(directory + '/' + "PPO_reward_data.txt", "w")
        while self.n_episodes < self.max_episodes:
            self.interact()  # 与环境交互（step一次）
            if self.n_episodes >= self.episodes_before_train:
                self.train_one_step()
            #  每训练1步计算一个reward
            if self.episode_done and ((self.n_episodes + 1) % 1 == 0):
                print("Training Episode %d" % (self.n_episodes + 1))
                th.save(self.actor.state_dict(), filename)
                test_reward = self.test_one_episode()
                print(test_reward)
                f.write(f"{test_reward}\n")
        f.close()
        self.plot_num_func()

    def test_one_episode(self):
        filename = "Pretrained_model/PPO_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)
        ep_reward = 0
        avg_reward = 0
        avg_num = 2
        state = self.env.reset()
        for ep in range(avg_num):
            for t in range(1, self.max_steps):
                action = self.action(state)  # 选择动作
                state, reward, done = self.env.step(action)
                ep_reward += reward
                if done:
                    break
            avg_reward += ep_reward
            ep_reward = 0
            ep_real_target = 0
            state = self.env.reset()
        avg_reward /= avg_num
        return avg_reward

    def test(self):
        filename = "Pretrained_model/PPO_model.pth"
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
                state, reward, done = self.env.step(action)
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

    def plot_num_func(self):
        reward_data = np.loadtxt('Pretrained_model/PPO_reward_data.txt')
        # reward_data = np.loadtxt('Pretrained_model/DDQN_saved_time_data.txt')
        reward_temp = 0
        reward_plot = []
        interval = 1  # 绘图间隔
        for i in range(reward_data.size):
            reward_temp += reward_data[i]
            if (i + 1) % interval == 0:
                reward_plot.append(reward_temp / interval)
                reward_temp = 0
        plt.figure()
        plt.plot(range(len(reward_plot)), reward_plot)
        label = ['PPO']
        plt.legend(label, loc='lower right')
        plt.savefig('./Pretrained_model/PPO.png')
        plt.show()


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out
