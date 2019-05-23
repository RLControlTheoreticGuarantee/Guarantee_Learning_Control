import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

def get_evaluation_rollouts(policy, env, num_of_paths, max_ep_steps, render= True):

    a_bound = env.action_space.high
    paths = []

    for ep in range(num_of_paths):
        s = env.reset()
        path = {'rewards':[],
                'lrewards':[]}
        for step in range(max_ep_steps):
            if render:
                env.render()
            a = policy.choose_action(s, evaluation=True)
            action = a * a_bound
            action = np.clip(action, -a_bound, a_bound)
            s_, r, done, info = env.step(action)
            l_r = info['l_rewards']

            path['rewards'].append(r)
            path['lrewards'].append(l_r)
            s = s_
            if done or step == max_ep_steps-1:
                paths.append(path)
                break
    if len(paths)< num_of_paths:
        print('no paths is acquired')

    return paths


def evaluate_rollouts(paths):
    total_returns = [np.sum(path['rewards']) for path in paths]
    total_lreturns = [np.sum(path['lrewards']) for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]
    import matplotlib.pyplot as plt
    [plt.plot(np.arange(0, len(path['rewards'])), path['rewards']) for path in paths]
    try:
        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('lreturn-average', np.mean(total_lreturns)),
            ('lreturn-min', np.min(total_lreturns)),
            ('lreturn-max', np.max(total_lreturns)),
            ('lreturn-std', np.std(total_lreturns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))
    except ValueError:
        print('Value error')
    else:
        return diagnostics


def evaluate_training_rollouts(paths):
    if len(paths) < 1:
        return None
    total_returns = [np.sum(path['rewards']) for path in paths]

    lyapunov_returns = [np.sum(path['l_rewards']) for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]
    l_error = [np.mean(path['l_error']) for path in paths]
    critic1_error = [np.mean(path['critic1_error']) for path in paths]
    critic2_error = [np.mean(path['critic2_error']) for path in paths]
    alpha = [np.mean(path['alpha']) for path in paths]
    labda = [np.mean(path['lambda']) for path in paths]
    log_pis = [np.mean(path['entropy']) for path in paths]
    a_loss = [np.mean(path['a_loss']) for path in paths]
    total_violations = [np.sum(path['violation']) for path in paths]

    diagnostics = OrderedDict((
        ('train-return-average', np.mean(total_returns)),
        ('train-lreturn-average', np.mean(lyapunov_returns)),
        ('train-episode-length-avg', np.mean(episode_lengths)),
        ('train-lyapunov-error-avg', np.mean(l_error)),
        ('train-critic1-error-avg', np.mean(critic1_error)),
        ('train-critic2-error-avg', np.mean(critic2_error)),
        ('train-alpha-avg', np.mean(alpha)),
        ('train-lambda-avg', np.mean(labda)),
        ('train-entropy-avg', np.mean(log_pis)),
        ('train-a-loss-avg', np.mean(a_loss)),
        ('train-violation-avg', np.mean(total_violations)),
    ))

    return diagnostics
