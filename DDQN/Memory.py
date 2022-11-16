
import random
from collections import namedtuple


Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        # # 如果超出了则去掉为0的
        # to_delete = []
        # if self.position == (self.capacity - 1):
        #     for ii in range(self.capacity):
        #         if self.memory[ii].rewards <= 0:
        #             to_delete.append(ii)
        #     for jj in to_delete[::-1]:
        #         del self.memory[jj]
        #         self.position -= 1
        self.position = (self.position + 1) % self.capacity


    def push(self, states, actions, Qs, next_states=None, dones=None):
        if isinstance(states, list):  # 如果states是list类型
            if next_states is not None and len(next_states) > 0:
                for s,a,r,n_s,d in zip(states, actions, Qs, next_states, dones):  # zip将对象中对应的元素打包成一个个元组
                    self._push_one(s, a, r, n_s, d)
            else:
                for s,a,r in zip(states, actions, Qs):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, Qs, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
