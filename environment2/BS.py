import math

import numpy as np
import Position
from environment2.DPUAV import DPUAV
from environment2.UAV import calcul_SNR


class BS:
    def __init__(self, position: Position, link_range: float):
        self.position = position
        """BS所在位置"""
        self.link_range = link_range
        """连接范围"""
        self.B_UAV = 1.0
        """与UAV之间的传输带宽"""

