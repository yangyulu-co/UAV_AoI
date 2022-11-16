from PPO import PPO
from environment.PlayGround import PlayGround
from utils import identity

from torch import nn

CRITIC_HIDDEN_SIZE = 32  # critic网络隐藏层数
CRITIC_LOSS = "mse"  # "mse" or "huber"
CRITIC_LR = 0.0002  # critic网络的learning rate
ACTOR_HIDDEN_SIZE = 32  # actor网络的隐藏层数
ACTOR_LR = 0.0002  # actor网络的learning rate
ACTOR_OUTPUT_ACT = nn.functional.log_softmax  # actor网络的输出层函数
OPTIMIZER_TYPE = "rmsprop"  # "rmsprop" or "adam"
MAX_GRAD_NORM = 0.5  # 梯度下降的最大梯度

TARGET_UPDATE_STEPS = 50  # 软更新频率
TARGET_TAU = 0.001  # 软更新比例

CLIP_PARAM = 0.2  # 软更新界限

ROLL_OUT_N_STEPS = 100  # roll out n steps
MEMORY_CAPACITY = ROLL_OUT_N_STEPS  # only remember the latest ROLL_OUT_N_STEPS
BATCH_SIZE = ROLL_OUT_N_STEPS  # only use the latest ROLL_OUT_N_STEPS for training A2C
MAX_STEPS = 100  # max steps in each episode, prevent from running too long
MAX_EPISODES = 10000  # 最大游戏次数
EPISODES_BEFORE_TRAIN = 1  # 开始训练之前游戏次数

REWARD_DISCOUNTED_GAMMA = 0.8  # reward衰减因子
DONE_PENALTY = None  # 结束游戏的惩罚
ENTROPY_REG = 0.00  # 熵的加权系数

# noisy exploration
EPSILON_START = 0.6
EPSILON_END = 0.4
EPSILON_DECAY = 200

USE_CUDA = False  # 是否使用GPU

# 设置环境
env = PlayGround()
state_dim = env.state_dim
action_dim = env.action_dim

# 创建类对象
ppo = PPO(env=env, state_dim=state_dim, action_dim=action_dim,
          memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
          roll_out_n_steps=ROLL_OUT_N_STEPS, episodes_before_train=EPISODES_BEFORE_TRAIN,
          reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
          batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
          actor_hidden_size=ACTOR_HIDDEN_SIZE, critic_hidden_size=CRITIC_HIDDEN_SIZE,
          actor_output_act=ACTOR_OUTPUT_ACT, critic_loss=CRITIC_LOSS,
          actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
          optimizer_type=OPTIMIZER_TYPE,
          epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
          epsilon_decay=EPSILON_DECAY, max_grad_norm=MAX_GRAD_NORM,
          target_tau=TARGET_TAU, target_update_steps=TARGET_UPDATE_STEPS,
          clip_param=CLIP_PARAM, use_cuda=USE_CUDA)

ppo.train()
ppo.test()