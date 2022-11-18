import numpy as np


class Model:
    """连续仿真的UAV场景下的AOI模型"""

    def __init__(self, num_UE, num_BS):
        self.num_UE = num_UE
        """用户(UE)数量"""
        self.num_BS = num_BS
        """基站(BS)数量"""
