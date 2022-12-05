import math

import numpy as np

from environment2.BS import BS
from environment2.UAV import UAV, calcul_channel_gain, calcul_SNR
from environment2.UE import UE

channel_gain = 1
"""BS和UAV之间的信道增益"""


class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self):
        self.B_ue = 5 * 10 ** 5
        """与ue之间的传输带宽"""

        self.transmission_power = None
        """无人机传输信号发射功率"""

    def get_transmission_rate_with_BS(self, bs: BS) -> float:
        """DPUAV和BS之间实际的传输速率"""
        SNR = calcul_SNR(self.transmission_power)
        gain = channel_gain
        return bs.B_UAV * math.log2(1 + gain * SNR)

    def get_transmission_time_with_BS(self, ue: UE, bs: BS) -> float:
        """传输单个ue任务到BS的时间"""
        rate = self.get_transmission_rate_with_BS(bs)
        return ue.task.storage / rate

    def get_transmission_energy_with_BS(self, ue: UE, bs: BS):
        """传输单个ue任务到BS的能耗"""
        energy = self.transmission_power * self.get_transmission_time_with_BS(ue, bs)
        return energy
