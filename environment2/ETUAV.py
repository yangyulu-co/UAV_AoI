from environment2.UAV import UAV, calcul_channel_gain


class ETUAV(UAV):
    """给UE进行无线充电的UAV，energy transmission"""

    def __init__(self):
        self.charge_power = 100
        """无人机无线充电的功率"""

    def charge_ue(self, ue):
        """给单个UE充电"""
        # 待定，判断是否在范围内
        gain = calcul_channel_gain(self.position, ue.position)
        ue.charge(gain * self.charge_power)


