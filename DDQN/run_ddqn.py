from DDQN import DDQN
from environment.UAV_UNKNOWN import UAV_Playground
from torch import nn


CRITIC_HIDDEN_SIZE = 32  # critic网络隐藏层数
CRITIC_LOSS = "mse"  # "mse" or "huber"
CRITIC_LR = 0.0001  # critic网络的learning rate
OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
MAX_GRAD_NORM = 0.5  # 梯度下降的最大梯度

MEMORY_CAPACITY = 100000  # memory可以存储的交互信息条数
MAX_STEPS = 10000  # max steps in each episode, prevent from running too long
MAX_EPISODES = 10000  # 最大游戏次数
EPISODES_BEFORE_TRAIN = 0  # 开始训练之前游戏次数
BATCH_SIZE = 100  # 训练时一次取的交互信息条数

REWARD_DISCOUNTED_GAMMA = 0.95  # reward衰减因子
DONE_PENALTY = None  # 结束游戏的惩罚

# noisy exploration
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 500

USE_CUDA = True  # 是否使用GPU

# 设置环境
env = UAV_Playground()
state_dim = env.state_dim
action_dim = env.action_dim

# 创建DDQN类对象
ddqn = DDQN(env=env, state_dim=state_dim, action_dim=action_dim,
            memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
            reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
            critic_hidden_size=CRITIC_HIDDEN_SIZE, critic_loss=CRITIC_LOSS,
            critic_lr=CRITIC_LR, optimizer_type=OPTIMIZER_TYPE,
            max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
            episodes_before_train=EPISODES_BEFORE_TRAIN,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, use_cuda=USE_CUDA
            )

ddqn.train()
ddqn.test()
