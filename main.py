from algorithm.DDPG import DDPG
from environment_cent.Area import Area
from common.utils import identity
import torch as th

CRITIC_HIDDEN_SIZE = 64  # critic网络隐藏层数
CRITIC_LOSS = "mse"  # "mse" or "huber"
CRITIC_LR = 0.001  # critic网络的learning rate
ACTOR_HIDDEN_SIZE = 64 # actor网络的隐藏层数
ACTOR_LR = 0.001  # actor网络的learning rate
ACTOR_OUTPUT_ACT = th.tanh # actor网络的输出层函数
OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
MAX_GRAD_NORM = 0.5  # 梯度下降的最大梯度

TARGET_UPDATE_STEPS = 5  # 软更新频率
TARGET_TAU = 0.01  # 软更新比例

MEMORY_CAPACITY = 1000  # memory可以存储的交互信息条数
MAX_STEPS = 100  # max steps in each episode, prevent from running too long
MAX_EPISODES = 20000  # 最大游戏次数
EPISODES_BEFORE_TRAIN = 10  # 开始训练之前游戏次数
BATCH_SIZE = 1000  # 训练时一次取的交互信息条数

REWARD_DISCOUNTED_GAMMA = 0.99  # reward衰减因子
DONE_PENALTY = 0.  # 结束游戏的惩罚

# noisy exploration
EPSILON_START = 0.5
EPSILON_END = 0.05
EPSILON_DECAY = 500

USE_CUDA = True  # 是否使用GPU

RANDOM_SEED = 2017

# 设置环境
env = Area()
state_dim = env.state_dim
action_dim = env.action_dim

# 创建DDPG类对象
ddpg = DDPG(env=env, state_dim=state_dim, action_dim=action_dim,
            memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
            target_tau=TARGET_TAU, target_update_steps=TARGET_UPDATE_STEPS,
            reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
            actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
            actor_output_act=ACTOR_OUTPUT_ACT, critic_loss=CRITIC_LOSS,
            actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,optimizer_type=OPTIMIZER_TYPE,
            max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
            episodes_before_train=EPISODES_BEFORE_TRAIN,
            epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
            epsilon_decay=EPSILON_DECAY, use_cuda=USE_CUDA)

ddpg.train()
ddpg.test()