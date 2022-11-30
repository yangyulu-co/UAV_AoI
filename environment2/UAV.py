import math

import numpy as np
import Position


def power_by_speed(v):
    """不同速度下的功率"""
    P0 = 0.012 / 8 * 1.225 * 0.05 * 0.503 * (300 ** 3) * (0.4 ** 3)
    U_tip = 120
    P1 = (1 + 0.1) * 20 ** (3 / 2) / math.sqrt(2 * 1.225 * 0.503)
    v0 = 4.03
    d0 = 0.6
    rho = 1.225
    s = 0.05
    A = 0.503
    P = P0 * (1 + (3 * v ** 2) / (U_tip ** 2)) + P1 * (
            math.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2)) ** 0.5
    + 0.5 * d0 * rho * s * A * v ** 3
    return P


class UAV:
    """UAV的基类"""

    def __init__(self, position, cover_distance, max_speed, transmit_power, height, speed_limit):
        self.position = position
        """UAV所在位置"""
        self.cover_distance = cover_distance
        """覆盖范围"""
        self.max_speed = max_speed
        """最大速度"""
        self.transmit_power = transmit_power
        """发射功率"""
        self.height = height
        """巡航高度"""

        self.speed_limit = speed_limit
        """速度限制"""
        self.time_slice = 1
        """一个时隙的时间长度"""

        self.energy_consumption = 0
        """累计无人机能量的消耗"""

    def distance(self, point):
        """与UE/BS之间的距离"""
        return self.position.distance(point.position)

    def if_connect(self, point):
        """UAV是否能连接到UE/BS"""
        return self.position.if_connect(point.position, point.link_range)

    # def move(self, x_move, y_move):
    #     """UAV位置的移动"""
    #     self.position.move(x_move, y_move)  # 更新位置

    def energy_by_speed(self, speed):
        """一个时隙内在恒定速度下的能耗"""
        return power_by_speed(speed) * self.time_slice

    def get_tail(self):
        """得到历史轨迹"""
        return self.position.tail

    def move_by_radian_rate(self, radian, rate):
        """无人机水平移动，rate参数为0到1之间的数"""
        if not 0 <= rate <= 1:
            print("移动速度超出限制")
            return False
        # 更新位置
        self.position.move_by_radian(radian, rate * self.speed_limit * self.time_slice)
        # 更新能耗
        self.energy_consumption += self.energy_by_speed(rate * self.speed_limit)
