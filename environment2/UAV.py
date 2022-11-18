import numpy as np
import Position


class UAV:
    def __init__(self, position, cover_distance):
        self.position = position
        """UAV所在位置"""



    def distance(self, point):
        """与UE/BS之间的距离"""
        return self.position.distance(point.position)

    def if_connect(self, point):
        """UAV是否能连接到UE/BS"""
        return self.position.if_connect(point.position, point.link_range)



