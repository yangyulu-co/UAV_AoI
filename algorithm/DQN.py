import os

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var, agg_double_list

import gym


class DQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000, max_episodes=2000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(DQN, self).__init__(env, state_dim, action_dim,
                                  memory_capacity, max_steps, max_episodes,
                                  reward_gamma, reward_scale, done_penalty,
                                  actor_hidden_size, critic_hidden_size,
                                  actor_output_act, critic_loss,
                                  actor_lr, critic_lr,
                                  optimizer_type, entropy_reg,
                                  max_grad_norm, batch_size, episodes_before_train,
                                  epsilon_start, epsilon_end, epsilon_decay,
                                  use_cuda)

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
        if self.use_cuda:
            self.actor.cuda()

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

    # agent interact with the environment to collect experience
    def interact(self):
        # ?????????????????????????????????????????????push???memory???
        super(DQN, self)._take_one_step()

    # train on a sample batch: ?????????memory???????????????batch???????????????actor???????????????????????????
    def train_one_step(self):
        if self.n_episodes <= self.episodes_before_train:
            pass
        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)  # view????????????reshape
        actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # ??????memory??????????????????states??????actor???network?????????Q(s_t)??????
        # gather?????????Q(s_t)????????????????????????????????????????????????Q(s_t, a)
        current_q = self.actor(states_var).gather(1, actions_var)
        # ???next states????????????actor???network?????????Q(s_{t+1})
        # ???Q(s_{t+1})??????????????????????????????next_q
        next_state_action_values = self.actor(next_states_var).detach()
        next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
        # compute target q by: r + gamma * max_a { Q(s_{t+1}) }
        target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)
        # update value network
        self.actor_optimizer.zero_grad()  # ???????????????
        # ????????????????????????loss
        if self.critic_loss == "huber":
            loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
        else:
            loss = th.nn.MSELoss()(current_q, target_q)
        # ??????????????????
        loss.backward()
        # ?????????????????????
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        # ??????????????????
        self.actor_optimizer.step()

    def train(self):
        while self.n_episodes < self.max_episodes:
            self.interact()  # ??????????????????step?????????
            if self.n_episodes >= self.episodes_before_train:
                self.train_one_step()  # ????????????actor???network
            if self.episode_done and ((self.n_episodes + 1) % 1 == 0):
                print("Training Episode %d" % (self.n_episodes + 1))

        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "DQN_model.pth"
        th.save(self.actor.state_dict(), filename)

    def test(self):
        filename = "Pretrained_model/DQN_model.pth"
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
    OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
    MAX_GRAD_NORM = 0.5  # ???????????????????????????

    MEMORY_CAPACITY = 10000  # memory?????????????????????????????????
    MAX_STEPS = 10000  # max steps in each episode, prevent from running too long
    MAX_EPISODES = 2000  # ??????????????????
    EPISODES_BEFORE_TRAIN = 0  # ??????????????????????????????
    BATCH_SIZE = 100  # ???????????????????????????????????????

    REWARD_DISCOUNTED_GAMMA = 0.99  # reward????????????
    DONE_PENALTY = -10.  # ?????????????????????

    # noisy exploration
    EPSILON_START = 0.5
    EPSILON_END = 0.05
    EPSILON_DECAY = 500

    USE_CUDA = True  # ????????????GPU

    RANDOM_SEED = 2017

    # ????????????
    env = gym.make("CartPole-v1")
    env.seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    if len(env.action_space.shape) > 1:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # ??????DQN?????????
    dqn = DQN(env=env, state_dim=state_dim, action_dim=action_dim,
              memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
              reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
              critic_hidden_size=CRITIC_HIDDEN_SIZE, critic_loss=CRITIC_LOSS,
              critic_lr=CRITIC_LR, optimizer_type=OPTIMIZER_TYPE,
              max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
              episodes_before_train=EPISODES_BEFORE_TRAIN,
              epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
              epsilon_decay=EPSILON_DECAY,use_cuda=USE_CUDA
              )

    dqn.train()
    dqn.test()


