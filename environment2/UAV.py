import math

import numpy as np
import Position
import environment2.Constant

def calcul_Prob_LoS(radian: float) -> float:
    """计算给定弧度下LoS的概率"""
    theta = math.degrees(radian)
    Prob_Los = 1 / (1 + 10 * math.exp(-0.6 * (theta - 10)))  # 城市环境中视距传输概率
    return Prob_Los


def calcul_Prob_hat(radian: float) -> float:
    """给定弧度下考虑到LOS信道的等效衰减系数"""
    chi = 0.2
    Prob_Los = calcul_Prob_LoS(radian)
    Prob_hat = Prob_Los * chi + (1 - Prob_Los)
    return Prob_hat


def calcul_channel_gain(position1: Position, position2: Position) -> float:
    """计算平均信道增益"""
    beta_0 = 1  # 待定
    alpha = 2.3
    radian = position1.downcast(position2)
    channel_gain = calcul_Prob_hat(radian) * (position1.distance(position2) ** -alpha) * beta_0
    return channel_gain


def calcul_SNR(p: float) -> float:
    sigma_2 = 1.0
    """计算信噪比"""
    return p / sigma_2


def power_by_speed(v: float) -> float:
    """不同速度下的功率,速度的单位为m/s,功率单位为W"""
    P0 = 0.012 / 8 * 1.225 * 0.05 * 0.503 * (300 ** 3) * (0.4 ** 3)
    U_tip = 120
    P1 = (1 + 0.1) * 20 ** (3 / 2) / math.sqrt(2 * 1.225 * 0.503)
    v0 = 4.03
    d0 = 0.6
    rho = 1.225
    s = 0.05
    A = 0.503
    P = P0 * (1 + (3 * v ** 2) / (U_tip ** 2)) + \
        P1 * (math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) ** 0.5 + \
        0.5 * d0 * rho * s * A * v ** 3
    return P


class UAV:
    """UAV的基类"""

    def __init__(self, position: Position, height: float, speed_limit: float):
        self.position = position
        """UAV所在位置"""

        self.height = height
        """巡航高度,单位m"""

        self.speed_limit = speed_limit
        """飞行最大速度，单位m/s"""


        self.energy_consumption = 0
        """累计无人机能量的消耗(Wh)"""

    def energy_by_speed(self, speed: float) -> float:
        """一个时隙内在恒定速度(m/s)下的能耗，单位为Wh"""
        return power_by_speed(speed) * (environment2.Constant.time_slice / (60 * 60))

    def get_tail(self):
        """得到历史轨迹"""
        return self.position.tail

    def move_by_radian_rate(self, radian: float, rate: float):
        """无人机水平移动，rate参数为0到1之间的数"""
        if not 0 <= rate <= 1:
            print("移动速度超出限制")
            return False
        # 更新位置
        self.position.move_by_radian(radian, rate * self.speed_limit * environment2.Constant.time_slice)
        # 更新能耗
        self.energy_consumption += self.energy_by_speed(rate * self.speed_limit)
