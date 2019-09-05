import numpy as np
from collections import OrderedDict


def get_evaluation_rollouts(policy, env, num_of_paths, max_ep_steps, render= True):

    a_bound = env.action_space.high
    paths = []

    for ep in range(num_of_paths):
        s = env.reset()
        path = {'rewards':[],
                'lrewards':[],
                'violation': [],}
        for step in range(max_ep_steps):
            if render:
                env.render()
            a = policy.choose_action(s, evaluation=True)
            action = a * a_bound
            action = np.clip(action, -a_bound, a_bound)
            s_, r, done, info = env.step(action)
            l_r = info['l_rewards']
            violation_of_constraint = info['violation_of_constraint']

            path['rewards'].append(r)
            path['lrewards'].append(l_r)
            path['violation'].append(violation_of_constraint)


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
    total_violations = [np.sum(path['violation']) for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]
    try:
        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('lreturn-average', np.mean(total_lreturns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('violation-avg', np.mean(total_violations)),

        ))
    except ValueError:
        print('Value error')
    else:
        return diagnostics


def evaluate_training_rollouts(paths):
    keys = paths[0].keys()
    summary = {}
    if len(paths) < 1:
        return None

    [summary.update({key:np.mean([np.mean(path[key]) for path in paths])}) for key in keys]
    summary['rewards'] = np.mean([np.sum(path['rewards']) for path in paths])
    summary['l_rewards'] = np.mean([np.sum(path['l_rewards']) for path in paths])
    summary['end_cost'] = np.mean([path['l_rewards'][-1] for path in paths])
    summary['violation'] = np.mean([np.sum(path['violation']) for path in paths])
    summary ['len']= np.mean([len(p['rewards']) for p in paths])


    return summary
