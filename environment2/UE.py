import numpy as np
import Position


class UE:
    def __init__(self, position, link_range):
        self.position = position
        """UE所在位置"""
        self.aoi = 0
        self.link_range = link_range
        """连接范围"""
        self.aoi_tail = [0]
        """历史aoi数据"""
    def distance(self, other_UE):
        """与其他节点的距离"""
        return self.position.distance(other_UE.position)

    def update_aoi(self, new_aoi):
        """更新AOI"""
        self.aoi = new_aoi
        self.aoi_tail.append(self.aoi)


