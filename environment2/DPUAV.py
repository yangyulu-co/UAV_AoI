import math

import numpy as np

from environment2.UAV import UAV, calcul_channel_gain


class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self):
        self.B = 5 * 10 ** 5  # 传输带宽

    def get_transmission_rate_with_UE(self, ue):
        """DPUAV和UE之间实际的传输速率"""
        gamma_0 = 10 ** (52.5 / 10)  # 信噪比乘上了beta0，此处由db单位转为数值，待定
        gain = calcul_channel_gain(self.position, ue.position)
        return self.B * math.log(1 + gain * gamma_0, 2)

    def get_transmission_time(self, ue):
        """传输ue任务到无人机的时间"""
        rate = self.get_transmission_rate_with_UE(ue)
        return ue.task.storage / rate
