import tensorflow as tf
import os
from variant import *
from disturber.disturber import Disturber
import numpy as np
import time
import logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_distrubance_function(env_name):
    if 'CartPolecons-v0' in env_name:
        disturbance_step = cartpole_disturber
    elif 'Pointcircle' in env_name:
        disturbance_step = point_circle_disturber
    elif 'Quadrotorcons' in env_name:
        disturbance_step = quadrotor_disturber
    else:
        disturbance_step = None
        print('no disturber designed for ' + env_name)
    return disturbance_step

def cartpole_disturber(time, s, action, env, eval_params):


    if time == eval_params['impulse_instant']:
        d = eval_params['magnitude'] * np.sign(s[0])
    else:
        d = 0

    s_, r, done, info = env.step(action, impulse=d)
    return s_, r, done, info

def point_circle_disturber(time, s, action, env, eval_params):


    if time == eval_params['impulse_instant']:
        d = eval_params['magnitude'] * np.sign(s[0])
    else:
        d = 0

    s_, r, done, info = env.step(action, impulse=d)
    return s_, r, done, info

def quadrotor_disturber(time, s, action, env, eval_params):


    if time == eval_params['impulse_instant']:
        d = eval_params['magnitude'] * np.sign(s[0])
    else:
        d = 0

    s_, r, done, info = env.step(action, impulse=d)
    return s_, r, done, info


def instant_impulse(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)
    env_params = variant['env_params']

    eval_params = variant['eval_params']
    policy_params = variant['alg_params']
    policy_params.update({
        's_bound': env.observation_space,
        'a_bound': env.action_space,
    })

    build_func = get_policy(variant['algorithm_name'])
    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0] \
                + env.observation_space.spaces['achieved_goal'].shape[0] + \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # d_dim = env_params['disturbance dim']
    policy = build_func(a_dim, s_dim, policy_params)
    # disturber = Disturber(d_dim, s_dim, disturber_params)

    log_path = variant['log_path'] + '/eval/safety_eval'
    variant['eval_params'].update({'magnitude': 0})
    logger.configure(dir=log_path, format_strs=['csv'])
    for magnitude in eval_params['magnitude_range']:
        variant['eval_params']['magnitude'] = magnitude
        diagnostic_dict = evaluation(variant, env, policy)

        string_to_print = ['magnitude', ':', str(magnitude), '|']
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        logger.logkv('magnitude', magnitude)
        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()



def evaluation(variant, env, policy, disturber= None):
    env_name = variant['env_name']
    # disturbance_step = get_distrubance_function(env_name)
    env_params = variant['env_params']

    max_ep_steps = env_params['max_ep_steps']

    eval_params = variant['eval_params']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    # For analyse
    Render = env_params['eval_render']

    # Training setting

    total_cost = []
    death_rates = []
    recovery_rate = []
    form_of_eval = variant['evaluation_form']
    trial_list = os.listdir(variant['log_path'])
    episode_length = []
    average_recovery_time = []

    for trial in trial_list:
        if trial == 'eval':
            continue
        if trial not in variant['trials_for_eval']:
            continue
        success_load = policy.restore(os.path.join(variant['log_path'], trial))
        if not success_load:
            continue
        seed_recovery_count = 0
        seed_average_cost = []
        seed_recovery_time = []
        for i in range(int(np.ceil(eval_params['num_of_paths']/(len(trial_list)-1)))):

            safety_cost = 0
            s = env.recovery_init(eval_params['magnitude'])
            if 'Fetch' in env_name or 'Hand' in env_name:
                s = np.concatenate([s[key] for key in s.keys()])
            a_path = []

            for j in range(max_ep_steps):
                if Render:
                    env.render()
                a = policy.choose_action(s, True)
                a_path.append(a)
                if variant['algorithm_name'] == 'LQR' or variant['algorithm_name'] == 'MPC':
                    action = a
                else:
                    action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
                # s_, r, done, info = disturbance_step(j, s, action, env, eval_params)
                s_, r, done, info = env.step(action)
                safety_cost += info['l_rewards']
                # if info['l_rewards'] <= 1e-4:
                #
                #     seed_recovery_count += 1
                #     seed_recovery_time.append(j)
                #     seed_average_cost.append(safety_cost)
                #     episode_length.append(j)
                #     break
                if 'Fetch' in env_name or 'Hand' in env_name:
                    s_ = np.concatenate([s_[key] for key in s_.keys()])
                    if info['done'] > 0:
                        done = True

                if j == max_ep_steps - 1:
                    done = True
                s = s_
                if done:
                    if info['l_rewards'] <= 1e-4:
                        seed_recovery_count += 1
                    seed_average_cost.append(safety_cost)
                    episode_length.append(j)
                    seed_recovery_time.append(j)
                    break
        recovery_rate.append(seed_recovery_count/(i+1) * 100)
        total_cost.append(np.mean(seed_average_cost))
        average_recovery_time.append((np.mean(seed_recovery_time)))

    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)

    average_length = np.average(episode_length)

    diagnostic = {'safety_return': total_cost_mean,
                  'safety_return_std': total_cost_std,
                  'recovery_time': np.average(average_recovery_time),
                  'recovery_time_std': np.std(average_recovery_time, axis=0),
                  'recovery_rate': np.average(recovery_rate),
                  'recovery_rate_std': np.std(recovery_rate, axis=0),
                  'average_length': average_length}
    return diagnostic




if __name__ == '__main__':
    for name in VARIANT['eval_list']:
        VARIANT['log_path'] = '/'.join(['./log', VARIANT['env_name'], name])

        if 'LAC' in name:
            VARIANT['alg_params'] = ALG_PARAMS['RLAC']
            VARIANT['algorithm_name'] = 'RLAC'
        elif'RARL' in name:
            VARIANT['alg_params'] = ALG_PARAMS['RARL']
            VARIANT['algorithm_name'] = 'RARL'
        elif 'SAC' in name:
            VARIANT['alg_params'] = ALG_PARAMS['SAC_cost']
            VARIANT['algorithm_name'] = 'SAC_cost'
        elif 'LQR' in name:
            VARIANT['alg_params'] = {}
            VARIANT['algorithm_name'] = 'LQR'
        elif 'MPC' in name:
            VARIANT['alg_params'] = ALG_PARAMS['MPC']
            VARIANT['algorithm_name'] = 'MPC'
        elif 'LSAC' in name:
            VARIANT['alg_params'] = ALG_PARAMS['LSAC']
            VARIANT['algorithm_name'] = 'LSAC'

        print('evaluating '+name)
        if VARIANT['evaluation_form'] == 'param_variation':
            param_variation(VARIANT)
        elif VARIANT['evaluation_form'] == 'trained_disturber':
            trained_disturber(VARIANT)
        else:
            instant_impulse(VARIANT)
        tf.reset_default_graph()

