from collections import OrderedDict, deque
import numpy as np
from copy import deepcopy

class Pool(object):

    def __init__(self, variant):

        s_dim = variant['s_dim']
        a_dim = variant['a_dim']
        self.memory_capacity = variant['memory_capacity']
        store_last_n_paths = variant['store_last_n_paths']
        self.paths = deque(maxlen=store_last_n_paths)
        self.reset()
        if 'history_horizon' in variant.keys():
            self.history_horizon = variant['history_horizon']
        else:
            self.history_horizon = 0
        self.memory = {
            's': np.zeros([self.history_horizon+1, s_dim]),
            'a': np.zeros([self.history_horizon+1, a_dim]),
            'r': np.zeros([self.history_horizon+1, 1]),
            'terminal': np.zeros([self.history_horizon+1, 1]),
            's_': np.zeros([self.history_horizon+1, s_dim]),

        }
        if 'finite_horizon' in variant.keys():
            if variant['finite_horizon']:
                self.memory.update({'value': np.zeros([self.history_horizon+1, 1])}),
                self.horizon = variant['value_horizon']
        self.memory_pointer = 0
        self.min_memory_size = variant['min_memory_size']

    def reset(self):
        [self.current_path.update({key: []}) for key in self.memory.keys()]

    def store(self, s, a, r, terminal, s_, kargs):
        transition = {'s': s, 'a': a, 'r': np.array([r]), 'terminal': np.array([terminal]), 's_': s_}
        if len(self.current_path['s']) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis,:]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate((self.current_path[key],transition[key][np.newaxis,:]))

        if terminal == 1.:
            if 'value' in self.memory.keys():
                r = deepcopy(self.current_path['r'])
                path_length = len(r)
                last_r = self.current_path['r'][-1, 0]
                r = np.concatenate((r,last_r*np.ones([self.horizon,1])), axis=0)
                value = []
                [value.append(r[i:i+self.horizon,0].sum()) for i in range(path_length)]
                value = np.array(value)
                self.memory['value'] = np.concatenate((self.memory['value'], value[:, np.newaxis]), axis=0)
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate((self.memory[key], self.current_path[key]), axis=0)
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory['s'])

        return self.memory_pointer

    def sample(self, batch_size):
        if self.memory_pointer < self.min_memory_size:
            return None
        else:
            indices = np.random.choice(min(self.memory_pointer,self.memory_capacity)-1, size=batch_size, replace=False) \
                      + max(1, 1+self.memory_pointer-self.memory_capacity)*np.ones([batch_size],np.int)
            batch = {}
            [batch.update({key: self.memory[key][indices]}) for key in self.memory.keys()]

            return batch


class Safe_Pool(Pool):

    def __init__(self, variant):
        super(Pool, self).__init__(variant)

        self.memory.update({'lr':np.zeros([1, s_dim])})
        if 'finite_safety_horizon' in variant.keys():
            if variant['finite_safety_horizon']:
                self.memory.update({'safety_value': np.zeros([1, 1])}),
            self.horizon = variant['safety_value_horizon']
        self.current_path = {}
        self.edge_memory_list = []
        self.temporal_edge_memory_list = []
        self.data_counter = 0

    def reset(self):
        [self.current_path.update({key: []}) for key in self.memory.keys()]


    def store(self, s, a, r,  terminal, s_, lr=0, edge_flag=False):
        transition = {'s': s, 'a': a, 'r': np.array([r]), 'lr': np.array([lr]), 'terminal': np.array([terminal]), 's_': s_}
        if len(self.current_path['s']) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis, :]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate((self.current_path[key], transition[key][np.newaxis, :]))

        if terminal == 1.:
            if 'value' in self.memory.keys():
                r = deepcopy(self.current_path['r'])
                path_length = len(r)
                last_r = self.current_path['r'][-1, 0]
                r = np.concatenate((r, last_r * np.ones([self.horizon, 1])), axis=0)
                value = []
                [value.append(r[i:i + self.horizon, 0].sum()) for i in range(path_length)]
                value = np.array(value)
                self.memory['value'] = np.concatenate((self.memory['value'], value[:, np.newaxis]), axis=0)

            if 'safety_value' in self.memory.keys():
                lr = deepcopy(self.current_path['lr'])
                path_length = len(lr)
                last_lr = self.current_path['lr'][-1, 0]
                lr = np.concatenate((lr, last_lr * np.ones([self.horizon, 1])), axis=0)
                safety_value = []
                [safety_value.append(lr[i:i + self.horizon, 0].sum()) for i in range(path_length)]
                safety_value = np.array(safety_value)
                self.memory['safety_value'] = np.concatenate((self.memory['safety_value'], safety_value[:, np.newaxis]), axis=0)

            for key in self.current_path.keys():
                self.memory[key] = np.concatenate((self.memory[key], self.current_path[key]), axis=0)
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory['s'])
            self.edge_memory_list += self.temporal_edge_memory_list
            self.temporal_edge_memory_list = []
        self.data_counter += 1
        if edge_flag:
            self.temporal_edge_memory_list.append(self.data_counter)
        return self.memory_pointer

    def sample(self, batch_size):
        if self.memory_pointer < self.min_memory_size:
            return None
        else:
            indices = np.random.choice(min(self.memory_pointer,self.memory_capacity)-1, size=batch_size, replace=False) \
                      + max(1, 1+self.memory_pointer-self.memory_capacity)*np.ones([batch_size],np.int)
            batch = {}
            [batch.update({key: self.memory[key][indices]}) for key in self.memory.keys()]

            return batch

class Robust_Pool(Pool):

    def __init__(self, variant):
        super(Pool, self).__init__(variant)
        self.memory.update({
            'd': np.zeros([1, d_dim]),
            'raw_d': np.zeros([1, d_dim]),})
        if variant['finite_safety_horizon']:
            self.memory.update({'safety_value':np.zeros([1, 1])}),
            self.horizon = variant['safety_value_horizon']
        self.current_path = {}

    def reset(self):
        [self.current_path.update({key: []}) for key in self.memory.keys()]


    def store(self, s, a, raw_d, r, terminal, s_):
        transition = {'s': s, 'a': a, 'd': d,'raw_d':raw_d, 'r': np.array([r]), 'terminal': np.array([terminal]), 's_': s_}
        if len(self.current_path['s']) < 1:
            for key in transition.keys():
                self.current_path[key] = transition[key][np.newaxis,:]
        else:
            for key in transition.keys():
                self.current_path[key] = np.concatenate((self.current_path[key],transition[key][np.newaxis,:]))

        if terminal == 1.:
            if 'value' in self.memory.keys():
                r = deepcopy(self.current_path['r'])
                path_length = len(r)
                last_r = self.current_path['r'][-1, 0]
                r = np.concatenate((r,last_r*np.ones([self.horizon,1])), axis=0)
                value = []
                [value.append(r[i:i+self.horizon,0].sum()) for i in range(path_length)]
                value = np.array(value)
                self.memory['value'] = np.concatenate((self.memory['value'], value[:, np.newaxis]), axis=0)
            for key in self.current_path.keys():
                self.memory[key] = np.concatenate((self.memory[key], self.current_path[key]), axis=0)
            self.paths.appendleft(self.current_path)
            self.reset()
            self.memory_pointer = len(self.memory['s'])

        return self.memory_pointer

    def sample(self, batch_size):
        if self.memory_pointer < self.min_memory_size:
            return None
        else:
            indices = np.random.choice(min(self.memory_pointer,self.memory_capacity)-1, size=batch_size, replace=False) \
                      + max(1, 1+self.memory_pointer-self.memory_capacity)*np.ones([batch_size],np.int)
            batch = {}
            [batch.update({key: self.memory[key][indices]}) for key in self.memory.keys()]

            return batch