import numpy as np
import Position


class BS:
    def __init__(self, position, link_range):
        self.position = position
        """BS所在位置"""
        self.link_range = link_range
        """连接范围"""

