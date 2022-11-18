import numpy as np
import Position


class UE:
    def __init__(self, position, link_range):
        self.position = position
        """UE所在位置"""
        self.aoi = 0
        self.link_range = link_range
        """连接范围"""

    def distance(self, other_UE):
        return self.position.distance(other_UE.position)


