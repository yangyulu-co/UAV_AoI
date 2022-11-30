import numpy as np
import Position


class UAV:
    """UAV的基类"""

    def __init__(self, position, cover_distance, max_speed, transmit_power, height, move_limit):
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

        self.move_limit = move_limit
        """每个时间间隔移动距离的限制，反应了无人机的移动速度"""

    def distance(self, point):
        """与UE/BS之间的距离"""
        return self.position.distance(point.position)

    def if_connect(self, point):
        """UAV是否能连接到UE/BS"""
        return self.position.if_connect(point.position, point.link_range)

    def move(self, x_move, y_move):
        """UAV位置的移动"""
        self.position.move(x_move, y_move)  # 更新位置

    def get_tail(self):
        """得到历史轨迹"""
        return self.position.tail

    def move_by_radian(self, radian, distance):
        """无人机水平移动，弧度形式"""
        if not 0 <= distance <= self.move_limit:
            print("移动距离超出限制")
            return False
        self.position.move_by_radian(radian, distance)

    def move_by_radian_rate(self, radian, rate):
        """无人机水平移动，rate参数为0到1之间的数"""
        self.move_by_radian(radian, self.move_limit * rate)
