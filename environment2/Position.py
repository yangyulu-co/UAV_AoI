import numpy as np


class Position:
    """在地图上的位置"""

    def __init__(self, x, y):
        self.data = np.empty((1, 2), np.float32)
        """位置的行向量"""
        self.data[0, 0] = x
        self.data[0, 1] = y
        self.tail = self.data.copy()
        """历史位置组成的数组，尺寸为n*2"""

    def distance(self, other_position):
        """两个位置之间的距离"""
        return np.linalg.norm(self.data - other_position.data)

    def print(self):
        """打印位置"""
        print(self.data)

    def if_connect(self, other_position, threshold):
        """两个位置之间的距离是否超过阈值"""
        return self.distance(other_position) <= threshold

    def move(self, x_move, y_move):
        """位置的移动"""
        # 保存新的位置
        self.data[0, 0] += x_move
        self.data[0, 1] += y_move
        # 将位置记录到历史位置中
        self.tail = np.vstack((self.tail, self.data))


if __name__ == "__main__":
    point1 = Position(3, 4)
    point1.print()
    point2 = Position(0, 0)
    print(point1.distance(point2))
    print(point1.if_connect(point2, 1))
    print(point1.if_connect(point2, 5))
    print(point1.if_connect(point2, 6))
    point1.move(1, 2)
    point1.move(1, 2)
    print(point1.tail)
