from matplotlib import pyplot as plt
import numpy as np

reward_data = np.loadtxt('Pretrained_model/DDQN_reward_data.txt')
# reward_data = np.loadtxt('Pretrained_model/DDQN_saved_time_data.txt')
reward_temp = 0
reward_plot = []
interval = 1  # 绘图间隔
for i in range(reward_data.size):
# for i in range(15000):
    reward_temp += reward_data[i]
    if (i + 1) % interval == 0:
        reward_plot.append(reward_temp / interval)
        reward_temp = 0

plt.figure()
plt.plot(range(len(reward_plot)), reward_plot)
label = ['A2C']
plt.legend(label, loc='lower right')
plt.savefig('./Pretrained_model/DDQN_threhold_reward.png')
# plt.savefig('./Pretrained_model/DDQN_threhold_saved_time.png')
plt.show()