import os

import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
# from torch.utils.tensorboard import SummaryWriter

import numpy as np
from matplotlib import pyplot as plt

from Memory import ReplayMemory
from utils import to_tensor_var, index_to_one_hot, entropy


class A2C:
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 roll_out_n_steps=10, max_episodes=2000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()

        self.memory = ReplayMemory(memory_capacity)
        self.roll_out_n_steps = roll_out_n_steps  # TD-n,交互n步后存入memory
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.n_episodes = 0
        self.n_steps = 0

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

        # self.writer = SummaryWriter()

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

        self.now_step = 0
        # self.step_record = []
        # self.loss_record = []
        self.episode_done = False

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

    def _take_n_steps(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)
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
        # dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

        # self.now_step += 1
        # self.step_record.append(self.now_step)
        # self.loss_record.append(np.mean((current_q.detach().cpu().numpy() - target_q.detach().cpu().numpy())) ** 2)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)  # actor网络输出概率(log)，batch_size * action_dim
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))  # 计算每次动作的熵并求平均
        action_log_probs = th.sum(action_log_probs * actions_var, 1)  # 以多大概率采取当前措施
        values = self.critic(states_var, actions_var)  # Critic网络计算当前V
        # A2C使用优势函数代替Critic网络中的原始回报，可以作为衡量选取动作值和所有动作平均值好坏的指标
        advantages = Qs_var - values.detach()  # 优势函数Q(s,a) - values
        pg_loss = -th.mean(action_log_probs * advantages)  # 计算优势函数的加权平均
        actor_loss = pg_loss - entropy_loss * self.entropy_reg  # 梯度-惩罚项（平均熵）
        self.now_step += 1
        # self.writer.add_scalar("Loss/train", actor_loss, self.now_step)
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
        directory = "Pretrained_model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + '/' + "A2C_model.pth"
        f = open(directory + '/' + "A2C_reward_data.txt", "w")
        f2 = open(directory + '/' + "A2C_saved_time_data.txt", "w")
        while self.n_episodes < self.max_episodes:
            self.interact()  # 与环境交互（step一次）
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
        filename = "Pretrained_model/A2C_model.pth"
        try:
            self.actor.load_state_dict(th.load(filename, map_location=lambda storage, loc: storage))
        except IOError as e:
            print(e)
        ep_real_target = 0
        ep_reward = 0
        avg_real_target = 0
        avg_reward = 0
        avg_num = 2
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
        self.fc1 = nn.Linear(state_dim, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = self.output_act(self.fc4(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.relu(self.fc3(out))
        out = self.fc4(out)
        return out
