# 作为暂时测试用
from environment_cent.DPUAV import DPUAV
from environment_cent.Position import Position
from environment_cent.Task import Task
from environment_cent.UE import UE

if __name__=='__main__':
    ue = UE(Position(0,0,0))
    ue.task = Task(5000,5)
    dpuav = DPUAV(Position(30,0,40))
    aoi_1 = dpuav.calcul_single_compute_and_offloading_aoi(ue,1)
    aoi_2 = dpuav.calcul_single_compute_and_offloading_aoi(ue,2)
    print(aoi_1,aoi_2)
    energy_1 = dpuav.calcul_single_compute_and_offloading_energy(ue,1)
    energy_2 = dpuav.calcul_single_compute_and_offloading_energy(ue,2)
    print(energy_1,energy_2)