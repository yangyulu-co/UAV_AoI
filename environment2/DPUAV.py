import math

import numpy as np

from environment2.UAV import UAV

# 计算与位置为w的用户之间传输时延, q,w: 1*2
def T_trans(q, w, radian):
    B = 5 * 10 ** 5  # 传输带宽
    gamma_0 = 10 ** (52.5 / 10)  # 信噪比相关参数
    temp = np.linalg.norm(q - w)  # q和w水平距离
    theta = 180 / math.pi *  radian  # q和w之间夹角
    Prob_Los = 1 / (1 + 10 * math.exp(-0.6 * (theta - 10)))  # 城市环境中视距传输概率
    Prob_hat = Prob_Los * 0.2 + (1 - Prob_Los)
    r = B * log(1 + gamma_0 * Prob_hat / ((temp ** 2 + H ** 2) ** (2.3 / 2)), 2)
    t = L / r
    return t

def calcul_Prob_LoS(radian):
    """计算给定弧度下LoS的概率"""
    theta = math.degrees(radian)
    Prob_Los = 1 / (1 + 10 * math.exp(-0.6 * (theta - 10)))  # 城市环境中视距传输概率
    return Prob_Los

def calcul_Prob_hat(radian):
    """给定弧度下考虑到LOS信道的等效衰减系数"""
    chi = 0.2
    Prob_Los = calcul_Prob_LoS(radian)
    Prob_hat = Prob_Los * chi + (1 - Prob_Los)
    return Prob_hat

def calcul_channel_gain(position1, position2):
    """计算平均信道增益"""
    beta_0 = 10 ** (52.5 / 10)  # 信噪比相关参数
    radian = position1.downcast(position2)
    channel_gain = calcul_Prob_hat(radian) * beta_0 * (position1.distance(position2) ** (-2.3 / 2))
    return channel_gain




class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self):

