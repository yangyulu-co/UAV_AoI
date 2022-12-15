import math

import numpy as np

from environment2.BS import BS
from environment2.Position import Position
from environment2.Task import Task
from environment2.UAV import UAV, calcul_channel_gain, calcul_SNR
from environment2.UE import UE



max_compute = 4
"""DP-UAV每个时刻最多能并行计算的用户数量"""

class DPUAV(UAV):
    """数据处理UAV,data process"""

    def __init__(self, position: Position):
        super().__init__(position, 100, 10)
        self.B_ue = 1 * (10 ** 6)
        """与ue之间的传输带宽(Hz)"""

        self.transmission_energy = 1*(10**(-3))
        """无人机传输信号发射能耗(j)"""

        self.computing_capacity = 5*(10**7)
        """DPUAV的计算能力，单位为cpu cycle/s"""

        self.link_range = 100.0
        """DPUAV和UE之间连接距离的限制，在此范围内才可以连接,单位为m"""

        self.rate_BS = 4*(10**(6))
        """与BS之间的通信速率(bit/s)"""

    def get_transmission_time_with_BS(self, ue: UE) -> float:
        """传输单个ue任务到BS的时间(s)"""
        return ue.task.storage / self.rate_BS

    def get_transmission_energy_with_BS(self, ue:UE)->float:
        """传输单个UE任务到BS消耗的功耗(J),待定"""
        return self.transmission_energy

    def get_compute_time(self, task:Task)->float:
        """任务所需要的计算时间(s)"""
        return task.compute / self.computing_capacity