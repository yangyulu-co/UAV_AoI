from DQN import DQN
from environment.UAV_PRO import UAV_Playground
from utils import identity

ACTOR_HIDDEN_SIZE = 32  # actor网络隐藏层数
ACTOR_LOSS = "mse"  # "mse" or "huber"
ACTOR_LR = 0.0001  # critic网络的learning rate
ACTOR_LR_END = 0.0001
OPTIMIZER_TYPE = "adam"  # "rmsprop" or "adam"
ACTOR_OUTPUT_ACT = identity  # identity or tanh
MAX_GRAD_NORM = 0.5  # 梯度下降的最大梯度
TARGET_TAU = 0.01  # 软更新参数
TARGET_UPDATE_STEP = 500  # 软更新频率

MEMORY_CAPACITY = 10000  # memory可以存储的交互信息条数
MAX_STEPS = 10000  # max steps in each episode, prevent from running too long
MAX_EPISODES = 10000  # 最大游戏次数
EPISODES_BEFORE_TRAIN = 0  # 开始训练之前游戏次数
BATCH_SIZE = 200  # 训练时一次取的交互信息条数

REWARD_DISCOUNTED_GAMMA = 0.85  # reward衰减因子
DONE_PENALTY = None  # 结束游戏的惩罚

# noisy exploration
EPSILON_START = 0.5
EPSILON_END = 0.1
EPSILON_DECAY = 500

USE_CUDA = True  # 是否使用GPU


# 设置环境
env = UAV_Playground()
state_dim = env.state_dim
action_dim = env.action_dim

# 创建DQN类对象
dqn = DQN(env=env, state_dim=state_dim, action_dim=action_dim,
          memory_capacity=MEMORY_CAPACITY, max_steps=MAX_STEPS, max_episodes=MAX_EPISODES,
          reward_gamma=REWARD_DISCOUNTED_GAMMA, done_penalty=DONE_PENALTY,
          target_tau=TARGET_TAU, target_update_steps=TARGET_UPDATE_STEP,
          actor_hidden_size=ACTOR_HIDDEN_SIZE, actor_output_act=ACTOR_OUTPUT_ACT, actor_loss=ACTOR_LOSS,
          actor_lr=ACTOR_LR, actor_lr_end=ACTOR_LR_END, optimizer_type=OPTIMIZER_TYPE,
          max_grad_norm=MAX_GRAD_NORM, batch_size=BATCH_SIZE,
          episodes_before_train=EPISODES_BEFORE_TRAIN,
          epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
          epsilon_decay=EPSILON_DECAY, use_cuda=USE_CUDA
          )

dqn.train()
dqn.test()
