import numpy as np
import Position


class UE:
    def __init__(self, position, link_range, lambda_high, lambda_low, speed_limit):
        self.position = position
        """UE所在位置"""
        self.aoi = 0
        """aoi"""
        self.link_range = link_range
        """连接范围"""
        self.aoi_tail = [0]
        """历史aoi数据"""

        self.labmda_high = lambda_high
        """高电量时产生数据的时间间隔(时隙数)"""
        self.labmda_low = lambda_low
        """低电量时产生数据的时间间隔(时隙数)"""

        self.energy = 100.0
        """用户的电量"""
        self.energy_max = 100.0
        """电量的最大值"""
        self.energy_threshold = 20.0
        """电量阈值，低于阈值，进入低功耗状态"""
        self.energy_state = 2
        """电量状态，2为高电量，1为低电量，0为无电"""
        self.energy_conversion_efficiency = 1
        """无线充电时能量收集效率"""

        self.next_generate = float('inf')
        """下一次产生数据的时间的倒计时"""
        self.buffer_if_full = False
        """是否存在生成好的任务"""
        self.task = None
        """生成好的任务"""

        self.speed_limit = speed_limit
        """速度限制"""
        self.time_slice = 1
        self.move_limit = self.speed_limit * self.time_slice
        """每个时间间隔移动距离的限制，反应了用户的移动速度"""

    def distance(self, other_UE):
        """与其他节点的距离"""
        return self.position.distance(other_UE.position)

    def update_energy_state(self):
        """更新电量状态"""
        if self.energy > self.energy_threshold:
            self.energy_state = 2
        elif 0 <= self.energy <= self.energy_threshold:
            self.energy_state = 1
        else:
            self.energy_state = 0

    def update_aoi(self, new_aoi):
        """更新AOI"""
        self.aoi = new_aoi
        self.aoi_tail.append(self.aoi)

    def offload_task(self):
        """UE卸载掉任务"""
        if not self.buffer_if_full:
            print("the ue don't have a task")
            return False
        self.buffer_if_full = False
        ans_task = self.task
        self.task = None
        return ans_task

    def move_by_radian(self, radian, distance):
        """用户水平移动，弧度形式"""
        if not 0 <= distance <= self.move_limit:
            print("移动距离超出限制")
            return False
        self.position.move_by_radian(radian, distance)

    def move_by_radian_rate(self, radian, rate):
        """用户水平移动，rate参数为0到1之间的数"""
        self.move_by_radian(radian, self.move_limit * rate)

    def charge(self, energy):
        """给UE充电"""
        temp_energy = energy * self.energy_conversion_efficiency
        energy = min(100.0, energy + temp_energy)
        self.update_energy_state()  # 更新电量状态




