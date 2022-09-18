from collections import namedtuple
import random
import numpy as np
import torch as th
import os
from tqdm import tqdm

names = ('states', 'actions', 'next_states', 'reward')
Experience = namedtuple('Experience', names)

root = './algo/files'


def read_file(filename):
    data = np.load(filename)

    length = len(data[names[0]])

    memory = []
    for i in tqdm(range(length), desc='Loading from {}'.format(filename)):
        args = [th.from_numpy(data[name][i]).float().data for name in names]
        memory.append(Experience(*args))
    return memory


class ReplayMemory:
    def __init__(self, capacity, release=False):
        self.capacity = capacity
        self.memory = {}
        self.position = {}

        if release:
            self.release()

    def push(self, *args):
        key = len(args[0])
        if key not in self.memory.keys():
            memory = self.memory[key] = []
            position = self.position[key] = 0
        else:
            memory = self.memory[key]
            position = self.position[key]

        if len(memory) < self.capacity:
            memory.append(None)

        memory[position] = Experience(*args)
        self.position[key] = int((position + 1) % self.capacity)

    def sample(self, batch_size, num_iter):
        for n, memory in self.memory.items():
            if len(memory) < batch_size:
                continue

            yield n, [random.sample(memory, batch_size) for _ in range(num_iter)]

    def counter(self):
        return {'macr_'+str(n): len(memory) for n, memory in self.memory.items()}

    def release(self):
        for file_or_folder in os.listdir(root):
            filename = os.path.join(root, file_or_folder)

            if os.path.isfile(filename) and file_or_folder.endswith('.npz'):
                n_agents = int(file_or_folder[:-4].split('_')[-1])
                memory = read_file(filename)
                print(n_agents, len(memory))
                if n_agents not in self.memory.keys():
                    self.memory[n_agents] = memory
                else:
                    self.memory[n_agents] += memory

        print([[n_agents, len(memory)] for n_agents, memory in self.memory.items()])

