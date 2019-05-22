


def get_safety_constraint_func(args):
    if args['env_name'] == 'cartpole':
        return judge_mujoco
    elif args['env_name']:
        return judge_mujoco

def judge_mujoco(s_, r, done, info):
    in_edge = False
    if info['l_rewards']>0.:
        in_edge = True

    return in_edge



def judge_cartpole(s_, r, done, info):
    in_edge = False
    if abs(s_[0]) > 0.8*info['cons_pos']:
        in_edge = True

    return in_edge
