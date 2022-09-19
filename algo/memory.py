from collections import namedtuple
import random
import numpy as np
import os

names = ('states', 'actions', 'next_states', 'reward')
Experience = namedtuple('Experience', names)

root = './algo/files'


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
        return {'macr_' + str(n): len(memory) for n, memory in self.memory.items()}

    def sample_(self, batch_size, num_iter):
        for n, [states, actions, n_states, rewards] in self.memory.items():
            length = len(states)
            if length < batch_size:
                continue

            experiences = []
            for _ in range(num_iter):
                idxes = random.sample(list(range(length)), batch_size)
                experiences.append((states[idxes], actions[idxes], n_states[idxes], rewards[idxes]))
            yield n, experiences

    def release(self):
        for file_or_folder in os.listdir(root):
            filename = os.path.join(root, file_or_folder)

            if os.path.isfile(filename) and file_or_folder.endswith('.npz'):
                n_agents = int(file_or_folder[:-4].split('_')[-1])
                # if n_agents not in [2, 4]:
                #     continue

                data = np.load(filename)
                print(n_agents, file_or_folder, [data[name].shape for name in names])

                # idx = np.where(data[names[-1]] >= 0)[0]
                if n_agents not in self.memory.keys():
                    # self.memory[n_agents] = [data[name][idx] for name in names]
                    self.memory[n_agents] = [data[name] for name in names]
                else:
                    memory = self.memory[n_agents]
                    # self.memory[n_agents] = [np.concatenate([memory[i], data[name][idx]])
                    #                          for i, name in enumerate(names)]
                    self.memory[n_agents] = [np.concatenate([memory[i], data[name]])
                                             for i, name in enumerate(names)]

                for n_agents, memory in self.memory.items():
                    print('\t', n_agents, [m.shape for m in memory])
