import tensorflow as tf
import os
from variant import VARIANT, get_env_from_name, get_policy, get_train
from robustness_eval import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ENV.env
if __name__ == '__main__':
    root_dir = VARIANT['log_path']
    if VARIANT['train']:
        for i in range(VARIANT['start_of_trial'], VARIANT['start_of_trial']+VARIANT['num_of_trials']):
            VARIANT['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + VARIANT['log_path'])
            train = get_train(VARIANT['algorithm_name'])
            train(VARIANT)

            tf.reset_default_graph()
    else:
        for name in VARIANT['eval_list']:
            VARIANT['log_path'] = '/'.join(['./log', VARIANT['env_name'], name])

            if 'LAC' in name:
                VARIANT['alg_params'] = ALG_PARAMS['RLAC']
                VARIANT['algorithm_name'] = 'RLAC'
            elif 'RARL' in name:
                VARIANT['alg_params'] = ALG_PARAMS['RARL']
                VARIANT['algorithm_name'] = 'RARL'
            elif 'SAC' in name:
                VARIANT['alg_params'] = ALG_PARAMS['SAC_cost']
                VARIANT['algorithm_name'] = 'SAC_cost'
            elif 'LQR' in name:
                VARIANT['alg_params'] = {}
                VARIANT['algorithm_name'] = 'LQR'
            else:
                VARIANT['alg_params'] = ALG_PARAMS['MPC']
                VARIANT['algorithm_name'] = 'MPC'
            print('evaluating ' + name)
            if VARIANT['evaluation_form'] == 'param_variation':
                param_variation(VARIANT)
            elif VARIANT['evaluation_form'] == 'trained_disturber':
                trained_disturber(VARIANT)
            else:
                instant_impulse(VARIANT)
            tf.reset_default_graph()

