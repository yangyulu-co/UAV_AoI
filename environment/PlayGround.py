import numpy as np
from matplotlib import pyplot as plt
from math import *

# from environment.User import User

X_MAX = 100
"""场地大小"""
grid_size = 20
"""网格大小"""
grid_num = int(X_MAX / grid_size)
"""网格数量"""
users_num = 16
"""用户数量"""
lam_low = 0
lam_high = 1  # 产生数据的参数（每个时隙预期产生数据数）
user_speed = 1
"""用户移动的最大速度"""
max_steps = 200
"""训练的次数（用来产生表示每个时隙是否采样的向量）"""


class PlayGround:
    def __init__(self):
        self.action_dim = 5  # action是上下左右停
        self.state_dim = 1 + 3 * users_num  # UAV位置，UE位置，UE的AoI，UAV端的AoI，

        self.UE_list = [User() for i in range(users_num)]
        """用户列表"""
        self.UE_loc, self.UE_lam, self.UE_AoI = extract_data(self.UE_list)  # 为了方便把用户的信息提取出来
        # print(self.UE_loc, self.UE_lam, self.UE_AoI)

        self.UAV_loc = 0  # UAV初始网格编号
        self.UAV_AoI = np.zeros(users_num)  # UAV端AoI

        self.env_state = np.concatenate(([self.UAV_loc], self.UE_loc, self.UE_AoI, self.UAV_AoI), axis=0)
        self.action = 0
        self.done = False  # 是否结束当前游戏
        self.reward = 0
        self.step_num = 0
        # print(self.env_state, self.reward, self.done)

    def reset(self):  # 返回env_state
        # # 重新产生用户
        # self.UE_list = []
        # for i in range(users_num):
        #     self.UE_list.append(User())
        # self.UE_loc, self.UE_lam, self.UE_AoI = extract_data(self.UE_list)  # 为了方便把用户的信息提取出来

        # 不重新产生用户
        self.UE_loc, self.UE_lam, self.UE_AoI = extract_data(self.UE_list)

        # 初始化UAV
        self.UAV_loc = 0  # UAV初始网格编号
        self.UAV_AoI = np.zeros(users_num)  # UAV端AoI

        self.env_state = np.concatenate(([self.UAV_loc], self.UE_loc, self.UE_AoI, self.UAV_AoI), axis=0)
        self.action = 0
        self.done = False  # 是否结束当前游戏
        self.reward = 0
        self.step_num = 0
        # print(self.env_state, self.reward, self.done)
        return self.env_state

    def step(self, action):  # 返回 next_state, reward, done, info
        self.reward = 0
        """UE端状态更新（位置和AoI）"""
        for user in self.UE_list:
            user.step_one_slot(self.step_num)
        self.UE_loc, _, self.UE_AoI = extract_data(self.UE_list)
        """UAV端状态更新（位置和AoI）"""
        self.action = action
        if self.action == 0:  # 上
            self.UAV_loc += grid_num
        elif self.action == 1:  # 下
            self.UAV_loc -= grid_num
        elif self.action == 2:  # 左
            self.UAV_loc -= 1
        elif self.action == 3:  # 右
            self.UAV_loc += 1
        self.UAV_loc = clamp(self.UAV_loc, 0, grid_num ** 2 - 1)
        # 计算UAV端的AoI
        self.UAV_AoI += np.ones_like(self.UAV_AoI)
        for j in range(users_num):
            if self.UAV_loc == self.UE_loc[j]:  # 连接上了j用户
                self.UAV_AoI[j] = self.UE_AoI[j]
        self.env_state = np.concatenate(([self.UAV_loc], self.UE_loc, self.UE_AoI, self.UAV_AoI), axis=0)
        """计算reward"""
        self.reward = - np.linalg.norm(self.UAV_AoI, ord=1)
        """是否结束"""
        if self.step_num >= max_steps - 1:
            self.done = True
        else:
            self.done = False
        self.step_num += 1
        # print(self.env_state, self.reward, self.done)
        return self.env_state, self.reward, self.done

    def render(self):  # 在屏幕上显示
        pass


class User:
    def __init__(self):
        self.location = np.random.uniform(0, X_MAX, 2)  # 用户位置
        self.grid_No = self.location[0] // grid_size + grid_num * (self.location[1] // grid_size)  # 网格编号
        self.grid_No = clamp(self.grid_No, 0, grid_num ** 2 - 1)
        self.lam = np.random.uniform(lam_low,lam_high)  # 用户产生数据的参数
        self.AoI = 0

        self.data_num_per_slot = np.random.poisson(lam=self.lam, size=max_steps)  # 每个时隙产生多少个数据服从泊松分布
        # print(self.data_num_per_slot)

    def step_one_slot(self, step_num):
        # # 移动
        # self.location += np.random.uniform(0, user_speed, 2)  # 用户随机移动
        # self.grid_No = self.location[0] // grid_size + grid_num * (self.location[1] // grid_size)  # 网格编号
        # self.grid_No = clamp(self.grid_No, 0, grid_num**2-1)
        # AoI变化
        if self.data_num_per_slot[step_num] == 0:
            self.AoI += 1  # 如果不产生数据AoI+1
        else:
            self.AoI = 0  # 如果产生数据AoI变为０


def extract_data(UE_list):
    """提取出用户信息"""
    UE_loc = [user.grid_No for user in UE_list] # 用户网格坐标
    UE_lam = [user.lam for user in UE_list] # 用户产生数据的参数
    UE_AoI = [user.AoI for user in UE_list] # 用户AoI
    return UE_loc, UE_lam, UE_AoI


def clamp(num, minn, maxn):  # 作用是限制num在[minn,maxn]内
    return max(min(num, maxn), minn)


if __name__ == "__main__":
    PG = PlayGround()
    PG.step(1)
