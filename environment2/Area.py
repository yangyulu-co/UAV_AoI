import random
from collections import defaultdict

import numpy as np

from environment2.Constant import N_user, N_ETUAV, N_DPUAV,eta_1,eta_2,eta_3
from environment2.DPUAV import DPUAV
from environment2.ETUAV import ETUAV
from environment2.Position import Position
from environment2.UE import UE


def get_link_dict(ues: [UE], dpuavs: [DPUAV]):
    """返回UEs和DAPUAVs之间的连接情况,返回一个dict,key为dpuav编号，value为此dpuav能够连接的ue组成的list"""

    link_dict = defaultdict(list)
    for i, ue in enumerate(ues):
        near_dpuav = None
        near_distance = None
        for j, dpuav in enumerate(dpuavs):
            if ue.if_link_DPUAV(dpuav):  # 如果在连接范围内
                distance = ue.distance_DPUAV(dpuav)
                if near_dpuav is None or near_distance > distance:
                    near_dpuav = j
                    near_distance = distance
        if near_distance is not None:
            link_dict[near_dpuav].append(i)

    return link_dict

def calcul_target_function(aois:[float],energy_dpuavs:[float],energy_etuavs:[float])->float:
    """计算目标函数的值"""
    return eta_1*sum(aois) + eta_2*sum(energy_dpuavs) + eta_3*sum(energy_etuavs)

class Area:
    """模型所在的场地范围"""

    def __init__(self, x_range=500.0, y_range=500.0):

        self.limit = np.empty((2, 2), np.float32)
        self.limit[0, 0] = -x_range / 2
        self.limit[1, 0] = x_range / 2
        self.limit[0, 1] = -y_range / 2
        self.limit[1, 1] = y_range / 2

        # 生成ue,etuav,dpuav
        self.UEs = self.generate_UEs(N_user)
        """所有ue组成的列表"""
        self.ETUAVs = self.generate_ETUAVs(N_ETUAV)
        """所有ETUAV组成的列表"""
        self.DPUAVs = self.generate_DPUAVs(N_DPUAV)
        """所有DPUAV组成的列表"""

        self.aoi = [0.0 for _ in range(N_user)]
        """UE的aoi"""

    def step(self, actions):
        # UE产生数据
        for ue in self.UEs:
            ue.generate_task()
        # 由强化学习控制，UAV开始运动
        etuav_move_energy = [0.0 for _ in range(N_ETUAV)]
        for i,etuav in enumerate(self.ETUAVs):
            etuav_move_energy[i] = etuav.move_by_radian_rate()
        dpuav_move_energy = [0.0 for _ in range(N_DPUAV)]
        for i,dpuav in enumerate(self.DPUAVs):
            dpuav_move_energy[i] = dpuav.move_by_radian_rate()

        # 计算连接情况
        link_dict = get_link_dict(self.UEs, self.DPUAVs)


        # 使用穷举方法，决定UAV的卸载决策






    def if_in_area(self, position) -> bool:
        """判断位置是否在场地里"""
        for i in range(2):
            if not self.limit[0, i] <= position.data[0, i] <= self.limit[1, i]:
                return False
        return True

    def generate_single_position(self) -> Position:
        """随机生成一个在区域里的点"""

        x = random.uniform(self.limit[0, 0], self.limit[1, 0])
        y = random.uniform(self.limit[0, 1], self.limit[1, 1])
        return Position(x, y, 0)

    def generate_positions(self, num: int) -> [Position]:
        """批量生成在区域里的点，返回一个列表"""
        return [self.generate_single_position() for _ in range(num)]

    def generate_UEs(self, num: int) -> [UE]:
        """生成指定数量的UE，返回一个list"""
        return [UE(self.generate_single_position()) for _ in range(num)]

    def generate_ETUAVs(self, num: int) -> [ETUAV]:
        """生成指定数量ETUAV，返回一个list"""
        return [ETUAV(self.generate_single_position()) for _ in range(num)]

    def generate_DPUAVs(self, num: int) -> [DPUAV]:
        """生成指定数量DPUAV，返回一个list"""
        return [DPUAV(self.generate_single_position()) for _ in range(num)]
