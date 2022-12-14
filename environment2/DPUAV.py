import math

import numpy as np

from environment2.BS import BS
from environment2.Position import Position
from environment2.UAV import UAV, calcul_channel_gain, calcul_SNR
from environment2.UE import UE

channel_gain = 1
"""BS和UAV之间的信道增益"""
energy_with_BS = 1
"""DP-UAV与基站通信需要的能量"""
max_connect = 4
"""DP-UAV每个时刻最多能并行计算的用户数量"""

class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self, position: Position):
        super().__init__(position, 100, 10)
        self.B_ue = 5 * 10 ** 5
        """与ue之间的传输带宽"""

        self.transmission_power = None
        """无人机传输信号发射功率"""

        self.computing_capacity = 5*(10**7)
        """DPUAV的计算能力，单位为cpu cycle/s"""

        self.link_range = 100.0
        """DPUAV和UE之间连接距离的限制，在此范围内才可以连接,单位为m"""

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

    def if_link(self, ue: UE) -> bool:
        """是否与UE连接"""
        return self.position.if_connect(ue.position, self.link_range)
