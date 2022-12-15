import random

import numpy as np
import math
import Position
from environment2.Task import Task
from environment2.UAV import calcul_SNR, calcul_channel_gain
from environment2.DPUAV import DPUAV


class UE:
    def __init__(self, position, link_range, lambda_high, lambda_low, speed_limit):
        self.position = position
        """UE所在位置"""
        self.aoi = 0.0
        """aoi"""
        self.link_range = link_range
        """连接范围"""
        self.aoi_tail = [0.0]
        """历史aoi数据"""

        self.lambda_high = lambda_high
        """高电量时产生数据的时间间隔(时隙数)"""
        self.lambda_low = lambda_low
        """低电量时产生数据的时间间隔(时隙数)"""
        self.high_probability = 0.5
        """高电量时每个时间间隔产生数据的概率，待定"""
        self.low_probability = 0.3
        """低电量时每个时间间隔产生数据的概率，待定"""

        self.energy = 1 * (10 ** (-5))
        """用户的电量(j)"""
        self.energy_max = 1 * (10 ** (-5))
        """电量的最大值"""
        self.energy_threshold = 2 * (10 ** (-6))
        """电量阈值，低于阈值，进入低功耗状态"""
        self.energy_state = 1
        """电量状态，1为高电量，0为低电量"""
        self.energy_conversion_efficiency = 0.1
        """无线充电时能量收集效率"""

        self.task = None
        """生成好的任务"""

        self.speed_limit = speed_limit
        """速度限制"""
        self.time_slice = 1
        self.move_limit = self.speed_limit * self.time_slice
        """每个时间间隔移动距离的限制，反应了用户的移动速度"""

        self.transmission_power = 1 * (10 ** (-5))
        """UE的发射功率(w)"""
        self.collect_energy = 5 * (10 ** (-7))
        """UE采集一个数据需要的能量(j)"""

    # 距离相关函数
    def distance(self, other_UE: 'UE') -> float:
        """与其他节点的距离"""
        return self.position.distance(other_UE.position)

    def if_link_DPUAV(self, uav: DPUAV) -> bool:
        """是否与UAV相连"""
        return self.position.if_connect(uav.position, uav.link_range)

    # 移动相关函数
    def move_by_radian(self, radian: float, distance: float):
        """用户水平移动，弧度形式"""
        if not 0 <= distance <= self.move_limit:
            print("移动距离超出限制")
            return False
        self.position.move_by_radian(radian, distance)

    def move_by_radian_rate(self, radian: float, rate: float):
        """用户水平移动，rate参数为0到1之间的数"""
        self.move_by_radian(radian, self.move_limit * rate)

    # 电量相关函数
    def update_energy_state(self):
        """更新电量状态"""
        if self.energy > self.energy_threshold:
            self.energy_state = 1
        else:
            self.energy_state = 0

    def charge(self, energy: float):
        """给UE充电，单位为J"""
        temp_energy = energy * self.energy_conversion_efficiency
        energy = min(self.energy_max, energy + temp_energy)
        self.update_energy_state()  # 更新电量状态

    def discharge(self, energy: float):
        """UE耗电，电量足够则扣除电量，返回True，否则不扣除电量并返回False"""
        if energy <= self.energy:
            self.energy -= energy
            self.update_energy_state()  # 更新电量状态
            return True
        return False

    # 传输相关函数
    def get_transmission_rate_with_UAV(self, uav: DPUAV) -> float:
        """DPUAV和UE之间实际的传输速率,单位为bit/s"""
        SNR = calcul_SNR(self.transmission_power)
        gain = calcul_channel_gain(uav.position, self.position)
        return uav.B_ue * math.log2(1 + gain * SNR)

    def get_transmission_time(self, uav: DPUAV) -> float:
        """UE传输单个任务到无人机的时间(s)"""
        rate = self.get_transmission_rate_with_UAV(uav)
        return self.task.storage / rate

    def get_transmission_energy(self, uav: DPUAV) -> float:
        """传输单个ue任务到无人机的能耗(J)"""
        energy = self.transmission_power * self.get_transmission_time(uav)
        return energy

    # 生成数据相关函数

    def generate_task(self):
        """每个时隙开始执行，按照电量产生数据并消耗能量"""
        generate = None  # 是否产生新数据
        if self.energy_state == 1:
            # 高电量
            generate = random.random() < self.high_probability
        else:
            # 低电量
            generate = random.random() < self.low_probability
        if generate:  # 如果要生成新数据
            if self.discharge(self.collect_energy):  # 如果电量足够并扣除电量
                self.task = Task()
                # 这里需要进一步处理

    def update_aoi(self, new_aoi: float):
        """更新AOI"""
        self.aoi = new_aoi
        self.aoi_tail.append(self.aoi)

    def offload_task(self):
        """UE卸载掉任务"""
        if self.task is None:
            print("the ue don't have a task")
            return False
        ans_task = self.task
        self.task = None
        return ans_task
