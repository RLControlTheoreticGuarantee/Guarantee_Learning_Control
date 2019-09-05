import gym
import datetime
import numpy as np
SEED = None
ITA = 1.
VARIANT = {
    'env_name': 'Pointcircle-v0',
    # 'env_name': 'CartPolecons-v0',
    # 'env_name': 'HalfCheetahcons-v0',
    'algorithm_name': 'LSAC',
    'additional_description': '-new-init-clip-80',
    'disturber': 'SAC',
    'evaluate': False,
    'train':True,
    'evaluation_frequency': 2048,
    'num_of_paths': 1,
    'num_of_trials': 5,
    'store_last_n_paths': 10,
    'start_of_trial': 2,

    # Evaluation Settings

    # 'evaluation_form': 'impulse',
    'evaluation_form': 'safety_eval',
    # 'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',
    'eval_list': [

        # 'MPC',
        # 'LQR',
        # 'LSAC-clip-20',
        'LSAC-new-init-clip-80',
        'LSAC-new-init-clip-50',
        'SSAC-new-init',
        'SAC-new-init',
        'LSAC-new-init',
        'SSAC',
        # 'SSAC',
        # 'SAC',
        # 'LAC',
        # 'RLAC',
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],
}
if VARIANT['algorithm_name'] == 'RARL':
    ITA = 0
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])
ENV_PARAMS = {
    'CartPolecons-v0': {
        'max_ep_steps': 250,
        'max_global_steps': int(6e5),
        'max_episodes': int(1e5),
        'eval_render': False,},
    'CartPolecost-v0': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e5),
        'disturbance dim': 1,
        'eval_render': False,},
    'Antcons-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(4e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'HalfCheetahcons-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e7),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Pointcircle-v0': {
        'max_ep_steps': 65,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Quadrotorcons-v0': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e7),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Quadrotorcost-v0': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,},

    'FetchReach-v1': {
        'max_ep_steps': 50,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'eval_render': False, },

    # 'Carcost-v0': {
    #     'max_ep_steps': 50,
    #     'max_global_steps': int(5e5),
    #     'max_episodes': int(1e6),
    #     'eval_render': False, },
     }
ALG_PARAMS = {

    'LSAC': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 0.8,
        'alpha': 1.,
        'alpha3':1.,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.99,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'target_entropy': None,
        'approx_value':True,
        'max_grad_norm': None,
        },
    'SAC': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 0.1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.99,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None,
        'max_grad_norm': None,
        },
    'SSAC': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'threshold':0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.99,
        'safety_gamma':0.5,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None,
        'max_grad_norm': None,
        },
    'CPO': {
        'batch_size':10000,
        'output_format':['csv'],
        'gae_lamda':0.95,
        'safety_gae_lamda':0.5,
        'labda': 1.,
        'alpha3': 0.1,
        'lr_c': 1e-4,
        'lr_l': 1e-4,
        'gamma': 0.995,
        'cliprange':0.2,
        'delta':0.01,
        'form_of_lyapunov': 'l_reward',
        'safety_threshold': 0.,
        'use_lyapunov': False,
        'use_adaptive_alpha3': False,
        'use_baseline':False,
        },
    'CPO_lyapunov': {
            'batch_size':10000,
            'output_format':['csv'],
            'gae_lamda':0.95,
            'safety_gae_lamda':0.5,
            'labda': 1.,
            'alpha3': 0.2,
            'lr_c': 1e-4,
            'lr_l': 1e-4,
            'gamma': 0.995,
            'cliprange':0.2,
            'delta':0.01,
            'form_of_lyapunov': 'l_value',
            'finite_horizon':True,
            'horizon': 5,
            'safety_threshold': 0.,
            'use_lyapunov': True,
            'use_adaptive_alpha3': False,
            'use_baseline':False,
            },
    'PDO': {
        'batch_size':1000,
        'output_format':['csv'],
        'gae_lamda':0.95,
        'safety_gae_lamda':0.5,
        'labda': 1.,
        'alpha3': 0.1,
        'lr_c': 1e-4,
        'lr_l': 1e-4,
        'gamma': 0.995,
        'cliprange':0.2,
        'delta':0.01,
        'form_of_lyapunov': 'l_reward',
        'safety_threshold': 10.,
        'use_lyapunov': False,
        'use_baseline':True,
        },
    'DDPG': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha3': 0.001,
        'tau': 5e-3,
        'noise': 0.1,
        'use_lyapunov': False,
        'lr_a': 3e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.99,
             },
    'LAC': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3':.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value':False,
        'target_entropy': None
    },
    'SAC_cost': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None
    },
    'MPC':{
        'horizon': 30,
        'control_horizon': 3,
    },

    'RLAC': {
        'iter_of_actor_train_per_epoch': 150,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 1.,
        'alpha3': 1.,
        'ita': ITA,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 5,
        'finite_horizon':False,
        'target_entropy': None
    },
    'RARL': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'ita': 0,
        'alpha': 1.,
        'alpha3': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 5,
        'train_per_cycle': 50,
        'use_lyapunov': False,
        'adaptive_alpha': True,
        'target_entropy': None
    },

}

DISTURBER_PARAMS = {

    'SAC': {
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'alpha': 1.,
        'ita': ITA,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'start_of_disturbance':0,
        'adaptive_alpha': True,
        'target_entropy': None,
        'energy_bounded': False,
        # 'energy_bounded': True,
        # 'process_noise': True,
        'process_noise': False,
        # 'noise_dim': 2,
        'energy_decay_rate': 0.5,
        # 'disturbance_magnitude': np.array([1]),
        'disturbance_magnitude': np.array([5, 0, 0, 0, 0]),
        # 'disturbance_magnitude': np.array([0.1, 1, 5,10]),
    },
}
EVAL_PARAMS = {
    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1

        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 100,   # number of path for evaluation
    },
    'impulse': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'magnitude_range': np.arange(10, 155, 5),
        'num_of_paths': 500,   # number of path for evaluation
    },
    'trained_disturber': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'path': './log/cartpole_cost/RLAC-full-noise-v2/0/',
        'num_of_paths': 100,   # number of path for evaluation

    },
    'safety_eval': {
        # 'magnitude_range': np.arange(80, 125, 5),
        'magnitude_range': np.arange(5, 45, 5), # point circle
        # 'magnitude_range': np.arange(4, 5.1, .1),  # cartpole_safe
        # 'magnitude_range': np.arange(3, 10, 1.),  # halfcheetah
        'num_of_paths': 500,   # number of path for evaluation

    },
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]
VARIANT['disturber_params']=DISTURBER_PARAMS[VARIANT['disturber']]
RENDER = True

def get_env_from_name(name):
    if name == 'Quadrotorcost-v0':
        env = gym.make('Quadrotorcons-v0')
        env = env.unwrapped
        env.modify_action_scale = False
        env.use_cost = True
    elif name == 'Carcost-v0':
        from ENV.env.classic_control.car_env import CarEnv
        env = CarEnv()

    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotorcons-v0':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    return env


def get_policy(name):
    if name == 'SAC'or name == 'LSAC':
        from LSAC.LSAC_V1 import SAC_with_lyapunov
        build_fn = SAC_with_lyapunov
    elif name=='SSAC':
        from SSAC.SSAC_V1 import SSAC
        build_fn = SSAC
    elif 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import LAC
        build_fn = LAC
    elif 'RARL' in name:
        from LAC.RARL import RARL as build_fn
    elif 'LQR' in name:
        from LQR.lqr import LQR as build_fn
    elif 'MPC' in name:
        from MPC.MPC import MPC as build_fn
    elif 'CPO' in name or 'PDO' in name:
        from CPO.CPO import CPO
        build_fn = CPO
    elif name == 'DDPG':
        from SAC.SRDDPG_V8 import DDPG
        build_fn = DDPG
    else:
        from LSAC.SAC_cost import SAC_cost as build_func
    return build_fn


def get_train(name):
    if name == 'SAC'or name == 'LSAC':
        from LSAC.LSAC_V1 import train
    elif name=='SSAC':
        from SSAC.SSAC_V1 import train
    elif 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import train
    if 'RARL' in name:
        from RARL.RARL import train as train
    elif 'CPO' in name or 'PDO' in name:
        from CPO.CPO import train

    elif name == 'DDPG':
        from LSAC.SRDDPG_V8 import train

    return train

