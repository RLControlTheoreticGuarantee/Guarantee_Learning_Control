import os
import time
import numpy as np
import os.path as osp
import logger
from collections import deque
from common import explained_variance, set_global_seeds
from common.policies_with_lyapunov import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from ppo2_lyapunov.runner import Runner


def constfn(val):
    def f(_):
        return val
    return f

def learn(*, network, env, total_timesteps,n_of_paths=10, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,lf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, alpha3 = .005, init_labda = 1.,labda_clip_range = 0.1,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2, use_adaptive_alpha3=True,approximate_value_function=False,
            save_interval=0, load_path=None, model_fn=None, **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)

    Parameters:
    ----------

    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                      specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                      tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                      neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                      See common/models.py/lstm for more details on using recurrent nets in policies

    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.


    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)

    total_timesteps: int              number of timesteps (i.e. number of actions taken in the environment)

    ent_coef: float                   policy entropy coefficient in the optimization objective

    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.

    vf_coef: float                    value function loss coefficient in the optimization objective

    max_grad_norm: float or None      gradient norm clipping coefficient

    gamma: float                      discounting factor

    lam: float                        advantage estimation discounting factor (lambda in the paper)

    log_interval: int                 number of timesteps between logging events

    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.

    noptepochs: int                   number of training epochs per update

    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training

    save_interval: int                number of timesteps between saving events

    load_path: str                    path to load the model from

    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.



    '''

    set_global_seeds(seed)
    logger.logkv('lr_a', lr)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    # 构建网络
    policy = build_policy(env, network, **network_kwargs)
    print("network build success")
    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ob_space_ = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps

    nbatch_train = nbatch // nminibatches

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from ppo2_lyapunov.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,lf_coef=lf_coef, ALPHA3= alpha3, init_labda=init_labda,
                    max_grad_norm=max_grad_norm, use_adaptive_alpha3=use_adaptive_alpha3,approximate_value_function=approximate_value_function)
    print("model build success")

    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    print("runner build success")
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam, n_of_paths=n_of_paths)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)



    logger.logkv('ent_coef', ent_coef)
    logger.logkv('vf_coef', vf_coef)
    logger.logkv('lf_coef', lf_coef)
    logger.logkv('max_grad_norm', max_grad_norm)
    logger.logkv('gamma', gamma)
    logger.logkv('advantage_lam', 0.95)

    logger.logkv('cliprange', cliprange)

    # Start total timer
    tfirststart = time.time()
    global_step = 0
    nupdates = total_timesteps//nbatch

    ep_L_R_threshold = 0
    ALPHA_MIN = 1000
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)
        # Get minibatch
        # 得到一连串的s,r,l_r,是否死了,a,v,l_v,mb_neglogpacs,s_,info
        obs,obs_, returns, l_returns,masks, actions, values, l_values,mb_l_rewards,neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632



        if eval_env is not None:
            eval_epinfos = eval_runner.eval_run() #pylint: disable=E0632

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, obs_, returns, l_returns, masks, actions, values, l_values, mb_l_rewards, neglogpacs))
                    # print(**slices)
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, obs_,returns,l_returns, masks, actions, values,l_values,mb_l_rewards, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.time()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        ep_L_R = safemean([epinfo['lr'] for epinfo in epinfobuf])
        if model.use_adaptive_alpha3:
            if ep_L_R > ep_L_R_threshold:
                ep_L_R_threshold = ep_L_R
                # model.ALPHA=min(model.ALPHA*1.05,0.1)  good  with out nothing and ini 10-5
                model.alpha3 = min(model.alpha3 * 1.5, labda_clip_range)
                # model.alpha3 = min(model.alpha3 * 1.1, labda_clip_range)

            # model.ALPHA = min(model.ALPHA * 1.001, 0.1)
            model.alpha3 = min(model.alpha3 * 1.01, labda_clip_range)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev_v = explained_variance(values, returns)
            ev_l = explained_variance(l_values, l_returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance_v", float(ev_v))
            logger.logkv("explained_variance_l", float(ev_l))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('violation_times', safemean([epinfo['violation'] for epinfo in epinfobuf]))
            logger.logkv('eplrewmean', safemean([epinfo['lr'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('alpha3', model.alpha3)
            if eval_env is not None:
                logger.logkv('eval_eprewmean', eval_epinfos['mean_rewards'] )
                logger.logkv('eval_eprewmin', eval_epinfos['min_rewards'])
                logger.logkv('eval_eprewmax', eval_epinfos['max_rewards'])
                logger.logkv('eval_eplrewmean', eval_epinfos['mean_lrewards'])
                logger.logkv('eval_eplrewmin', eval_epinfos['min_lrewards'])
                logger.logkv('eval_eplrewmax', eval_epinfos['max_lrewards'])
                logger.logkv('eval_eplenmean', eval_epinfos['mean_length'])
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and (MPI is None or MPI.COMM_WORLD.Get_rank() == 0):
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            # savepath = 'Model/1.ckpt'
            print('Saving to', savepath)
            model.save(savepath)
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)



