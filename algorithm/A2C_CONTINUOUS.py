import os

import torch as th
from torch import nn
from torch.distributions import MultivariateNormal
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import to_tensor_var, identity
import gym


class A2C(Agent):
    """
    An agent learned with Advantage Actor-Critic
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    """

    def __init__(self, env, state_dim, action_dim,
                 action_std_init=1., min_action_std=0.1,
                 action_std_decay_rate=0.05, action_std_decay_freq=500,
                 memory_capacity=10000, max_steps=1000,
                 roll_out_n_steps=10, max_episodes=2000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(A2C, self).__init__(env, state_dim, action_dim,
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
        self.action_std_2 = th.full((self.action_dim,), self.action_std * self.action_std)  # ????????????
        self.action_std_decay_rate = action_std_decay_rate
        self.action_std_decay_freq = action_std_decay_freq
        self.min_action_std = min_action_std

        self.roll_out_n_steps = roll_out_n_steps  # TD-n,??????n????????????memory

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    # ??????????????????????????????????????????
    def decay_action_std(self):
        self.action_std = self.action_std - self.action_std_decay_rate
        self.action_std = round(self.action_std, 2)
        if self.action_std <= self.min_action_std:
            self.action_std = self.min_action_std
        self.action_std_2 = th.full((self.action_dim,), self.action_std * self.action_std)

    # ?????????????????????epsilon-greedy???
    def exploration_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action_std_2_var = to_tensor_var(self.action_std_2, self.use_cuda)
        action_mean = self.actor(state_var)  # ????????????????????????
        cov_mat = th.diag_embed(action_std_2_var).unsqueeze(dim=0)  # ??????
        dist = MultivariateNormal(action_mean, cov_mat)  # ??????????????????
        action = dist.sample()  # ???dist???????????????
        return action.numpy().flatten()

    # ?????????????????????????????????
    def action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        action = self.actor(state_var)  # ????????????????????????
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
        # TD-n,??????n????????????????????????memory
        super(A2C, self)._take_n_steps()

    # train on a sample batch: ?????????memory???????????????batch???????????????actor???????????????????????????
    def train_one_step(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.action_dim)
        Qs_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_mean = self.actor(states_var)  # ????????????????????????
        action_std_2_var = to_tensor_var(self.action_std_2.expand_as(action_mean), self.use_cuda)
        cov_mat = th.diag_embed(action_std_2_var).unsqueeze(dim=0)  # ??????
        dist = MultivariateNormal(action_mean, cov_mat)  # ??????????????????
        action_log_probs = dist.log_prob(actions_var)  # ?????????????????????????????????????????????batch_size????????????
        entropy_loss = th.mean(dist.entropy())

        values = self.critic(states_var, actions_var)  # Critic??????????????????V
        # A2C????????????????????????Critic???????????????????????????????????????????????????????????????????????????????????????????????????
        advantages = Qs_var - values.detach()  # ????????????Q(s,a) - values
        pg_loss = -th.mean(action_log_probs * advantages)  # ?????????????????????????????????
        actor_loss = pg_loss - entropy_loss * self.entropy_reg  # ??????-????????????????????????
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = Qs_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

    def train(self):
        while self.n_episodes < self.max_episodes:
            self.interact()  # ??????????????????step?????????
            if self.n_episodes >= self.episodes_before_train:
                self.train_one_step()  # ????????????actor???network
            if self.episode_done and ((self.n_episodes + 1) % 100 == 0):
                print("Training Episode %d" % (self.n_episodes + 1))

        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "A2C2_model.pth"
        if os.path.exists(filename):
            os.remove(filename)
        th.save(self.actor.state_dict(), filename)

    def test(self):
        filename = "Pretrained_model/A2C2_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)

        test_running_reward = 0
        for ep in range(1, 11):
            ep_reward = 0
            state = self.env.reset()
            for t in range(1, 400 + 1):
                action = self.action(state)  # ????????????
                state, reward, done, _ = self.env.step(action)
                ep_reward += reward
                self.env.render()  # ??????????????????
                if done:
                    break
            test_running_reward += ep_reward
            print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))


if __name__ == "__main__":
    CRITIC_HIDDEN_SIZE = 32  # critic??????????????????
    CRITIC_LOSS = "mse"  # "mse" or "huber"
    CRITIC_LR = 0.001  # critic?????????learning rate
    ACTOR_HIDDEN_SIZE = 32  # actor?????????????????????
    ACTOR_LR = 0.001  # actor?????????learning rate
    ACTOR_OUTPUT_ACT = identity  # actor????????????????????????
    OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
    MAX_GRAD_NORM = 0.5  # ???????????????????????????

    ROLL_OUT_N_STEPS = 100  # roll out n steps
    MEMORY_CAPACITY = ROLL_OUT_N_STEPS  # only remember the latest ROLL_OUT_N_STEPS
    BATCH_SIZE = ROLL_OUT_N_STEPS  # only use the latest ROLL_OUT_N_STEPS for training A2C
    MAX_STEPS = 10000  # max steps in each episode, prevent from running too long
    MAX_EPISODES = 2000  # ??????????????????
    EPISODES_BEFORE_TRAIN = 100  # ??????????????????????????????

    REWARD_DISCOUNTED_GAMMA = 0.99  # reward????????????
    DONE_PENALTY = -10.  # ?????????????????????
    ENTROPY_REG = 0.00  # ??????????????????

    # noisy exploration
    ACTION_STD_INIT = 1.  # ???????????????
    MIN_ACTION_STD = 0.1  # ???????????????
    ACTION_STD_DECAY_RATE = 0.05  # ?????????????????????
    ACTION_STD_DECAY_FREQ = 500  # ?????????????????????

    USE_CUDA = True  # ????????????GPU

    RANDOM_SEED = 2017

    # ????????????
    env = gym.make("Pendulum-v1")
    env.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ??????A2C?????????
    a2c = A2C(env=env, state_dim=state_dim, action_dim=action_dim,
              action_std_init=ACTION_STD_INIT, min_action_std=MIN_ACTION_STD,
              action_std_decay_rate=ACTION_STD_DECAY_RATE, action_std_decay_freq=ACTION_STD_DECAY_FREQ,
              memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
              roll_out_n_steps=ROLL_OUT_N_STEPS, episodes_before_train=EPISODES_BEFORE_TRAIN,
              reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
              batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
              actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
              actor_output_act=ACTOR_OUTPUT_ACT, critic_loss=CRITIC_LOSS,
              actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
              optimizer_type=OPTIMIZER_TYPE,
              max_grad_norm=MAX_GRAD_NORM,
              use_cuda=USE_CUDA)

    a2c.train()
    a2c.test()
