import gym
import datetime
SEED = None
VARIANT = {
    'env_name': 'cartpole_cost',
    'algorithm_name': 'LAC',
    # 'algorithm_name': 'SAC_cost',
    'additional_description': '-value-perturb',
    'evaluate': False,
    'train':False,
    'evaluation_frequency': 2048,
    'num_of_paths': 1,
    'num_of_trials': 500,
    'store_last_n_paths': 10,
    'start_of_trial': 0,
}
VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])
ENV_PARAMS = {
    'cartpole': {
        'max_ep_steps': 250,
        'max_global_steps': int(6e5),
        'max_episodes': int(6e5),
        'eval_render': False,},
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e5),
        'impulse_mag':85,
        'eval_render': False,},
    'Antcpo-v1': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(4e6),
        'eval_render': False,},
    'HalfCheetah-v4': {
        'max_ep_steps': 200,
        'max_global_steps': int(6e5),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'PointCircle-v1': {
        'max_ep_steps': 65,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Quadrotor-v1': {
        'max_ep_steps': 2000,
        'max_global_steps': int(3e6),
        'max_episodes': int(1e6),
        'eval_render': False,},
    'Quadrotor-v1_cost': {
        'max_ep_steps': 2000,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,},

    'FetchReach-v1': {
        'max_ep_steps': 50,
        'max_global_steps': int(3e5),
        'max_episodes': int(1e6),
        'eval_render': False, },

    'car_env': {
        'max_ep_steps': 50,
        'max_global_steps': int(5e5),
        'max_episodes': int(1e6),
        'eval_render': False, },

     }
ALG_PARAMS = {
    'LSAC': {
        'memory_capacity': int(1e6),
        'cons_memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
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
        'safety_threshold': 10.,
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
            'form_of_lyapunov': 'l_reward',
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
        'alpha3':1.,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 50,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value':True,
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
}
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]
RENDER = True
def get_env_from_name(name):
    if name == 'cartpole':
        from envs.ENV_V0 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'car_env':
        from envs.car_env import CarEnv
        env = CarEnv()

    elif name == 'Quadrotor-v1_cost':
        env = gym.make('Quadrotor-v1')
        env = env.unwrapped
        env.modify_action_scale = False
        env.use_cost = True
    else:
        env = gym.make(name)
        env = env.unwrapped
        if name == 'Quadrotor-v1':
            if 'CPO' not in VARIANT['algorithm_name']:
                env.modify_action_scale = False
        if 'Fetch' in name or 'Hand' in name:
            env.unwrapped.reward_type = 'dense'
    env.seed(SEED)
    return env


def get_policy(name):
    if name == 'SAC'or name == 'LSAC':
        from SAC.SAC_V1 import SAC_with_lyapunov
        build_fn = SAC_with_lyapunov
    elif name=='SSAC':
        from SSAC.SSAC_V1 import SSAC
        build_fn = SSAC
    elif 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import LAC
        build_fn = LAC
    elif 'CPO' in name or 'PDO' in name:
        from CPO.CPO import CPO
        build_fn = CPO
    elif name == 'DDPG':
        from SAC.SRDDPG_V8 import DDPG
        build_fn = DDPG
    return build_fn


def get_train(name):
    if name == 'SAC'or name == 'LSAC':
        from SAC.SAC_V1 import train
    elif name=='SSAC':
        from SSAC.SSAC_V1 import train
    elif 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import train
    elif 'CPO' in name or 'PDO' in name:
        from CPO.CPO import train

    elif name == 'DDPG':
        from SAC.SRDDPG_V8 import train

    return train


def get_eval(name):
    if 'LAC' in name or 'SAC_cost' in name:
        from LAC.LAC_V1 import eval

    return eval


