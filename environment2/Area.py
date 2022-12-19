import random
from collections import defaultdict
from copy import copy

import numpy as np

from environment2.Constant import N_user, N_ETUAV, N_DPUAV, eta_1, eta_2, eta_3
from environment2.DPUAV import DPUAV, max_compute
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


def calcul_target_function(aois: [float], energy_dpuavs: [float], energy_etuavs: [float]) -> float:
    """计算目标函数的值"""
    return eta_1 * sum(aois) + eta_2 * sum(energy_dpuavs) + eta_3 * sum(energy_etuavs)


def generate_solution(ue_num: int) -> list:
    """根据输入的UE数量，返回所有的可行的卸载决策"""
    max_count = 3 ** ue_num
    possible_solutions = []
    for i in range(max_count):
        code = [0 for _ in range(ue_num)]
        for j in range(ue_num):
            code[j] = (i // (3 ** j)) % 3
        if code.count(1) <= max_compute:
            possible_solutions.append(code)

    return possible_solutions


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
        """ETUAV运动的能耗"""
        for i, etuav in enumerate(self.ETUAVs):
            etuav_move_energy[i] = etuav.move_by_radian_rate()
        dpuav_move_energy = [0.0 for _ in range(N_DPUAV)]
        """DPUAV运动的能耗"""
        for i, dpuav in enumerate(self.DPUAVs):
            dpuav_move_energy[i] = dpuav.move_by_radian_rate()

        # 计算连接情况
        link_dict = get_link_dict(self.UEs, self.DPUAVs)

        # 使用穷举方法，决定UAV的卸载决策
        offload_choice = self.find_best_offload(link_dict)
        sum_dpuav_energy = sum(dpuav_move_energy)
        """DPUAV总的能耗"""
        sum_etuav_energy = sum(etuav_move_energy)
        """ETUAV总的能耗"""
        offload_energy = [0.0 for _ in range(N_user)]
        offload_aoi = [self.aoi[i] + 1 for i in range(N_user)]
        for dpuav_index, ue_index, choice in offload_choice:
            energy, aoi = self.calcul_single_dpuav_single_ue_energy_aoi(dpuav_index, ue_index, choice)
            offload_energy[ue_index] = energy
            offload_aoi[ue_index] = aoi
        sum_dpuav_energy += sum(offload_energy)
        sum_aoi = sum(offload_aoi)
        target = eta_1 * sum_aoi + eta_2 * sum_dpuav_energy + eta_3 * sum_etuav_energy

        # ETUAV充电
        for etuav in self.ETUAVs:
            etuav.charge_all_ues(self.UEs)

    def calcul_single_dpuav_single_ue_energy_aoi(self, dpuav_index: int, ue_index: int, offload_choice):
        """计算单个dpuav单个ue的卸载决策下的能量消耗和aoi"""
        energy = self.DPUAVs[dpuav_index].calcul_single_compute_and_offloading_energy(self.UEs[ue_index],
                                                                                      offload_choice)
        aoi = self.DPUAVs[dpuav_index].calcul_single_compute_and_offloading_aoi(self.UEs[ue_index], offload_choice)
        if aoi is None:
            aoi = self.aoi[ue_index] + 1
        return energy, aoi

    def find_single_dpuav_best_offload(self, dpuav_index: int, ue_index_list: list):
        """穷举查找单个DPUAV下多个用户的最优卸载决策,返回数据格式为[dpuav_index,ue_index,{0,1,2}]组成的list"""
        solutions = generate_solution(len(ue_index_list))
        best_target = float('inf')
        best_solution = None

        for solution in solutions:
            solution_energy = 0.0
            solution_aoi = 0.0

            for i in range(len(ue_index_list)):
                energy, aoi = self.calcul_single_dpuav_single_ue_energy_aoi(dpuav_index, ue_index_list[i], solution[i])
                solution_energy += energy
                solution_aoi += aoi
            target = solution_energy * eta_2 + solution_aoi * eta_1

            if target < best_target:
                best_solution = copy(solution)
                best_target = target

        ans = []
        for i in range(len(ue_index_list)):
            ans.append([dpuav_index, ue_index_list[i], best_solution[i]])
        return ans

    def find_best_offload(self, link: dict):
        """穷举查找多个DPUAV下多个用户的最优卸载决策,返回数据格式为[dpuav_index,ue_index,{0,1,2}]组成的list"""
        ans = []
        for dpuav in link.keys():
            single_ans = self.find_single_dpuav_best_offload(dpuav, link[dpuav])
            ans += single_ans
        return ans

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


if __name__ == "__main__":
    print(generate_solution(1))
    print('----')
    print(generate_solution(2))
    print('----')
    print(generate_solution(3))
    print('----')
    print(generate_solution(4))
    print('----')
