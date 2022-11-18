import numpy as np
import Position

class Area:
    """模型所在的场地范围"""

    def __init__(self, x_min, x_max, y_min, y_max):

        self.limit = np.empty((2,2), np.float32)
        self.limit[0,0] = x_min
        self.limit[1,0] = x_max
        self.limit[0,1] = y_min
        self.limit[1,1] = y_max

    def if_in_area(self, position):
        """判断位置是否在场地里"""
        for i in range(2):
            if not self.limit[0, i] <= position.data[0,i] <= self.limit[1, i]:
                return False
        return True

