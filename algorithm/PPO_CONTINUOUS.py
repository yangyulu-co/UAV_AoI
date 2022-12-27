import os

import torch as th
from matplotlib import pyplot as plt
from torch import nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import to_tensor_var, index_to_one_hot, identity, agg_double_list
import gym


class PPO(Agent):

    def __init__(self, env, state_dim, action_dim,
                 action_std_init=1., min_action_std=0.1,
                 action_std_decay_rate = 0.05, action_std_decay_freq = 100,
                 memory_capacity=10000, max_steps=None, max_episodes=2000,
                 roll_out_n_steps=1, target_tau=1.,
                 clip_param=0.2, target_update_steps=5,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=th.tanh, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(PPO, self).__init__(env, state_dim, action_dim,
                                  memory_capacity, max_steps, max_episodes,
                                  reward_gamma, reward_scale, done_penalty,
                                  actor_hidden_size, critic_hidden_size,
                                  actor_output_act, critic_loss,
                                  actor_lr, critic_lr,
                                  optimizer_type, entropy_reg,
                                  max_grad_norm, batch_size, episodes_before_train,
                                  epsilon_start, epsilon_end, epsilon_decay,
                                  use_cuda)

        self.action_std = action_std_init
        self.action_std_2 = th.full((self.action_dim,), self.action_std * self.action_std)  # 初始方差
        self.action_std_decay_rate = action_std_decay_rate
        self.action_std_decay_freq = action_std_decay_freq
        self.min_action_std = min_action_std

        self.roll_out_n_steps = roll_out_n_steps  # TD-n,交互n步后存入memory
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        # self.actor_loss_record = []

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    # 连续状态下用来衰减分布的方差
    def decay_action_std(self):
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 2)
        if self.action_std <= self.min_action_std:
            self.action_std = self.min_action_std
        self.action_std_2 = th.full((self.action_dim,), self.action_std * self.action_std)

    # 选择一个动作（epsilon-greedy）
    def exploration_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action_std_2_var = to_tensor_var(self.action_std_2, self.use_cuda)
        action_mean = self.actor(state_var)  # 网络输出的平均值
        cov_mat = th.diag_embed(action_std_2_var).unsqueeze(dim=0)  # 方差
        dist = MultivariateNormal(action_mean, cov_mat)  # 多元正态分布
        action = dist.sample()  # 从dist分布中采样
        return action.numpy().flatten()

    # 选择一个动作（无探索）
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = self.actor(state_var)  # 网络输出的平均值
        return action.detach().numpy().flatten()

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    # agent interact with the environment to collect experience
    def interact(self):
        # TD-n,交互n步后将每一步存入memory
        super(PPO, self)._take_n_steps()

    # train on a sample batch: 执行从memory中提取一个batch数据，并对actor的网络进行一次更新
    def train_one_step(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.action_dim)
        Qs_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()

        values_pridict = self.critic_target(states_var, actions_var).detach()  # Critic_target网络计算当前V
        advantages = (Qs_var - values_pridict).view(1, -1)  # 优势函数
        action_mean = self.actor(states_var)  # 网络输出的平均值
        action_std_2_var = to_tensor_var(self.action_std_2.expand_as(action_mean), self.use_cuda)
        cov_mat = th.diag_embed(action_std_2_var).unsqueeze(dim=0)  # 方差
        dist = MultivariateNormal(action_mean, cov_mat)  # 多元正态分布

        old_action_mean = self.actor_target(states_var)
        old_dist = MultivariateNormal(old_action_mean, cov_mat)
        action_log_probs = dist.log_prob(actions_var)  # 每一步以多大概率采取当前措施（batch_size维向量）
        old_action_log_probs = old_dist.log_prob(actions_var)  # actor_target以多大概率采取当前措施
        # dist_entropy = dist.entropy()

        ratio = th.exp(action_log_probs - old_action_log_probs)  # 重要性采样的系数
        surr1 = ratio * advantages  #
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages  # 给重要性采样的系数限定一个范围
        actor_loss = -th.mean(th.min(surr1, surr2))  # 重要性采样过后优势函数的平均
        # self.actor_loss_record = self.actor_loss_record.append(actor_loss.detatch().numpy())

        actor_loss.mean().backward()
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
            super(PPO, self)._soft_update_target(self.actor_target, self.actor)
            super(PPO, self)._soft_update_target(self.critic_target, self.critic)

    def train(self):
        episodes = []
        eval_rewards = []
        while self.n_episodes <= self.max_episodes:
            self.interact()  # 与环境交互（step一次）
            if self.n_episodes >= self.episodes_before_train:
                self.train_one_step()  # 更新一次actor的network
            if self.episode_done and (self.n_episodes+1) % self.action_std_decay_freq == 0:
                self.decay_action_std()

            if self.episode_done and ((self.n_episodes + 1) % 100 == 0):
                print("Training Episode %d" % (self.n_episodes + 1))
                env_eval = gym.make("Pendulum-v1")
                rewards, _ = ppo.evaluation(env_eval)
                rewards_mu, rewards_std = agg_double_list(rewards)
                print("Episode %d, Average Reward %.2f" % (ppo.n_episodes + 1, rewards_mu))
                episodes.append(ppo.n_episodes + 1)
                eval_rewards.append(rewards_mu)

        episodes = np.array(episodes)
        eval_rewards = np.array(eval_rewards)
        plt.figure()
        plt.plot(episodes, eval_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(["PPO"])
        plt.savefig("ppo.png" )

        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "PPO2_model.pth"
        if os.path.exists(filename):
            os.remove(filename)
        th.save(self.actor.state_dict(), filename)

    def test(self):
        filename = "Pretrained_model/PPO2_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)

        test_running_reward = 0
        for ep in range(1, 11):
            ep_reward = 0
            state = self.env.reset()
            for t in range(1, 400 + 1):
                action = self.action(state)  # 选择动作
                state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                env.render()  # 在屏幕上显示
                if done:
                    break
            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))


if __name__ == "__main__":
    CRITIC_HIDDEN_SIZE = 32  # critic网络隐藏层数
    CRITIC_LOSS = "mse"  # "mse" or "huber"
    CRITIC_LR = 0.001  # critic网络的learning rate
    ACTOR_HIDDEN_SIZE = 32  # actor网络的隐藏层数
    ACTOR_LR = 0.001  # actor网络的learning rate
    ACTOR_OUTPUT_ACT = th.tanh  # actor网络的输出层函数
    OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
    MAX_GRAD_NORM = 0.5  # 梯度下降的最大梯度

    TARGET_UPDATE_STEPS = 5  # 软更新频率
    TARGET_TAU = 0.01  # 软更新比例

    CLIP_PARAM = 0.2  # 软更新界限

    ROLL_OUT_N_STEPS = 100  # roll out n steps
    MEMORY_CAPACITY = ROLL_OUT_N_STEPS  # only remember the latest ROLL_OUT_N_STEPS
    BATCH_SIZE = ROLL_OUT_N_STEPS  # only use the latest ROLL_OUT_N_STEPS for training A2C
    MAX_STEPS = 10000  # max steps in each episode, prevent from running too long
    MAX_EPISODES = 2000  # 最大游戏次数
    EPISODES_BEFORE_TRAIN = 100  # 开始训练之前游戏次数

    REWARD_DISCOUNTED_GAMMA = 0.99  # reward衰减因子
    DONE_PENALTY = -10.  # 结束游戏的惩罚
    ENTROPY_REG = 0.00  # 熵的加权系数

    # noisy exploration
    ACTION_STD_INIT = 0.6  # 初始标准差
    ACTION_STD_DECAY_RATE = 0.1  # linearly decay action_std 标准差下降幅度
    ACTION_STD_DECAY_FREQ = 200  # 标准差下降频率
    MIN_ACTION_STD = 0.1  # 最小标准差

    USE_CUDA = True  # 是否使用GPU

    RANDOM_SEED = 2022

    # 设置环境
    env = gym.make("BipedalWalker-v3")
    env.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 创建类对象
    ppo = PPO(env=env, state_dim=state_dim, action_dim=action_dim,
              action_std_init=ACTION_STD_INIT, action_std_decay_rate=ACTION_STD_DECAY_RATE,
              action_std_decay_freq = ACTION_STD_DECAY_FREQ, min_action_std = MIN_ACTION_STD,
              max_steps=MAX_STEPS, memory_capacity=MEMORY_CAPACITY,
              roll_out_n_steps=ROLL_OUT_N_STEPS, batch_size=BATCH_SIZE,
              episodes_before_train=EPISODES_BEFORE_TRAIN, max_episodes=MAX_EPISODES,
              entropy_reg=ENTROPY_REG, done_penalty=DONE_PENALTY,
              actor_output_act=ACTOR_OUTPUT_ACT, critic_loss=CRITIC_LOSS,
              actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
              actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, optimizer_type=OPTIMIZER_TYPE,
              target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
              reward_gamma=REWARD_DISCOUNTED_GAMMA,
              max_grad_norm=MAX_GRAD_NORM, clip_param=0.2,
              use_cuda=True)

    ppo.train()
    ppo.test()
