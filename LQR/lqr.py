import numpy as np
import math
import scipy.linalg as linalg
lqr = linalg.solve_continuous_are
import time
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
import logger
from variant import *


class LQR(object):
    def __init__(self, a_dim, d_dim, s_dim, variant):
        theta_threshold_radians = 20 * 2 * math.pi / 360
        length = 0.5
        masscart = 1
        masspole = 0.1
        total_mass = (masspole + masscart)
        polemass_length = (masspole * length)
        g = 10
        H = np.array([
            [1, 0, 0, 0],
            [0, total_mass, 0, - polemass_length],
            [0, 0, 1, 0],
            [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
        ])

        Hinv = np.linalg.inv(H)

        A = Hinv @ np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, - polemass_length * g, 0]
        ])
        B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
        Q = np.diag([1/100, 0., 20 *(1/ theta_threshold_radians)**2, 0.])
        R = np.array([[0.1]])

        P = lqr(A, B, Q, R)
        Rinv = np.linalg.inv(R)
        K = Rinv @ B.T @ P
        # H = np.mat([[masscart+masspole, masspole*length],
        #             [masspole*length,   masspole*length**2]])
        # dev_G = np.mat([[0,0],
        #                 [0,-masspole*g*length]])
        # A_sup = np.concatenate([np.zeros([2,2]),
        #                         np.diag(np.ones([2]))], axis=1)
        # A_sub = np.concatenate([-H.I*dev_G,np.zeros([2,2])],axis=1)
        # A = np.concatenate([A_sup,A_sub],axis=0)
        # B_dev = np.mat([[0],
        #                 [1]])
        # B = np.concatenate([np.zeros([2,1]),
        #                     H.I*B_dev],axis=0)
        #
        # Q = np.diag([0.1, 1.0, 100.0, 5.0])
        #
        # R = np.mat([1.])
        # K = np.mat(np.ones([1,4]))
        # P = np.mat(np.zeros([4,4]))
        # P_piao = np.mat(np.diag(np.ones([4])))
        # i = 0
        # ### optimize the controler
        # while np.linalg.norm(P_piao-P)>1e-8:
        #     P = P_piao
        #     K = -(R+gamma*B.T*P*B).I*B.T*P*A
        #     P_piao = Q + K.T*R*K + gamma * (A+B*K).T * P *(A+B*K)
        #     i += 1

        self.K = K

    def choose_action(self, x, arg):
        x1 = np.copy(x)
        x1[2] = np.sin(x1[2])
        return np.dot(self.K, x1)
    def restore(self, log_path):

        return True

def eval(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    env_params = variant['env_params']


    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']


    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    root_path = variant['log_path']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = policy_build_fn(policy_params)
    if 'cartpole' in env_name:
        mag = env_params['impulse_mag']
    # For analyse
    Render = env_params['eval_render']
    # Training setting
    t1 = time.time()
    die_count = 0
    for i in range(variant['num_of_paths']):

        log_path = variant['log_path']+'/eval/' + str(0)
        logger.configure(dir=log_path, format_strs=['csv'])
        s = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            s = np.concatenate([s[key] for key in s.keys()])

        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            if (j+1)%100 ==0 and 'cartpole'in env_name:

                impulse = mag * np.sign(s[0])
                # print('impulse comming:',impulse)
            # Run in simulator
                s_, r, done, info = env.step(action,impulse=impulse)
            else:
                s_, r, done, info = env.step(action)
            if 'Fetch' in env_name or 'Hand' in env_name:
                s_ = np.concatenate([s_[key] for key in s_.keys()])
                if info['done'] > 0:
                    done = True
            logger.logkv('rewards', r)
            logger.logkv('timestep', j)
            logger.dumpkvs()
            l_r = info['l_rewards']
            if j == max_ep_steps - 1:
                done = True
            s = s_
            if done:
                if j < 199:

                    die_count+=1
                if 'cartpole' in env_name:
                    print('episode:', i,
                          'death:', die_count,
                          'mag:',mag
                          )
                break
    print('Running time: ', time.time() - t1)
    return

if __name__=='__main__':
    lqr_policy = LQR(0.95)
