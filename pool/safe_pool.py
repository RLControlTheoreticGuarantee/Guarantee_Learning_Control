from collections import OrderedDict, deque
import numpy as np
from copy import deepcopy
from pool import Pool

class Safe_Pool(Pool):

    def __init__(self, variant):
        super(Pool, self).__init__(variant)

        self.memory.update({'lr':np.zeros([1, s_dim])})
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


