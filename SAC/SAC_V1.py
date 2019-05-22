
import tensorflow as tf
import numpy as np
import time
from SAC.squash_bijector import SquashBijector
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
from variant import VARIANT, get_env_from_name, get_policy, get_train
from .utils import get_evaluation_rollouts, evaluate_rollouts, evaluate_training_rollouts
import logger
from safety_constraints import get_safety_constraint_func

SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 50)

class SAC_with_lyapunov(object):
    def __init__(self,
                 a_dim,
                 s_dim,

                 variant,

                 action_prior = 'uniform',
                 ):



        ###############################  Model parameters  ####################################
        self.memory_capacity = variant['memory_capacity']
        self.cons_memory_capacity = variant['cons_memory_capacity']
        self.batch_size = variant['batch_size']
        gamma = variant['gamma']
        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        self.max_grad_norm = variant['max_grad_norm']
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 3), dtype=np.float32)
        self.cons_memory = np.zeros((self.cons_memory_capacity, s_dim * 2 + a_dim + 3), dtype=np.float32)
        self.pointer = 0
        self.cons_pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        self.a_dim, self.s_dim, = a_dim, s_dim,
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim  #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.cons_S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.cons_S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input')
        self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.l_R = tf.placeholder(tf.float32, [None, 1], 'l_r')  # 给lyapunov设计的reward
        self.cons_l_R = tf.placeholder(tf.float32, [None, 1], 'cons_l_r')
        self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal')
        self.LR_A = tf.placeholder(tf.float32, None, 'LR_A')
        self.LR_C = tf.placeholder(tf.float32, None, 'LR_C')
        self.LR_L = tf.placeholder(tf.float32, None, 'LR_L')
        # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
        labda = variant['labda']
        alpha = variant['alpha']
        alpha3 = variant['alpha3']
        log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
        log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature
        self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
        self.alpha = tf.exp(log_alpha)

        self.a, self.deterministic_a, self.a_dist = self._build_a(self.S, )  # 这个网络用于及时更新参数
        self.q1 = self._build_c(self.S, self.a_input, 'critic1')  # 这个网络是用于及时更新参数
        self.q2 = self._build_c(self.S, self.a_input, 'critic2')  # 这个网络是用于及时更新参数
        self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络

        self.q1_a = self._build_c(self.S, self.a, 'critic1', reuse=True)
        self.q2_a = self._build_c(self.S, self.a, 'critic2', reuse=True)

        self.use_lyapunov = variant['use_lyapunov']
        self.adaptive_alpha = variant['adaptive_alpha']

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic1')
        c2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic2')
        l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lyapunov')

        ###############################  Model Learning Setting  ####################################
        ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement
        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))
        target_update = [ema.apply(a_params), ema.apply(c1_params),ema.apply(c2_params), ema.apply(l_params)]  # soft update operation

        # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        a_, _, a_dist_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        cons_a, _, cons_a_dist = self._build_a(self.cons_S, reuse=True)
        cons_a_, _, cons_a_dist_ = self._build_a(self.cons_S_, reuse=True)
        self.cons_a_input = tf.placeholder(tf.float32, [None, a_dim], 'cons_a_input')
        # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
        # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
        self.log_pis = log_pis = self.a_dist.log_prob(self.a)
        self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

        # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q1_ = self._build_c(self.S_, a_,'critic1', reuse=True, custom_getter=ema_getter)
        q2_ = self._build_c(self.S_, a_, 'critic2', reuse=True, custom_getter=ema_getter)
        l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
        self.cons_l = self._build_l(self.cons_S, self.cons_a_input, reuse=True)
        self.cons_l_ = self._build_l(self.cons_S_, cons_a_, reuse=True)

        # lyapunov constraint

        self.l_derta = tf.reduce_mean(self.cons_l_ - self.cons_l + alpha3 * self.cons_l_R)

        labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))
        self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(alpha_loss, var_list=log_alpha)
        self.lambda_train = tf.train.AdamOptimizer(self.LR_A).minimize(labda_loss, var_list=log_labda)
        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim),
                scale_diag=tf.ones(self.a_dim))
            policy_prior_log_probs = policy_prior.log_prob(self.a)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        min_Q_target = tf.reduce_min((self.q1_a, self.q2_a), axis=0)
        self.a_preloss = a_preloss = tf.reduce_mean(self.alpha * log_pis - min_Q_target - policy_prior_log_probs)


        if self.use_lyapunov is True:
            a_loss = self.labda * self.l_derta + a_preloss
        else:
            a_loss = a_preloss
        self.a_loss = a_loss
        self.trainer = tf.train.AdamOptimizer(self.LR_A)
        pre_grads_and_var = self.trainer.compute_gradients(a_preloss, a_params)

        pre_grads, pre_var = zip(*pre_grads_and_var)
        # self.pre_max_grad = tf.reduce_max(tf.concat([tf.reshape(grad, [-1]) for grad in pre_grads], axis=0))

        grads_and_var = self.trainer.compute_gradients(a_loss, a_params)
        grads, var = zip(*grads_and_var)
        # self.max_grad = tf.reduce_max(tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0))

        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            pre_grads, _grad_norm = tf.clip_by_global_norm(pre_grads, self.max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

        grads_and_var = list(zip(grads, var))
        pre_grads_and_var = list(zip(pre_grads, pre_var))

        self.atrain = self.trainer.apply_gradients(grads_and_var) #以learning_rate去训练，方向是minimize loss，调整列表参数，用adam
        self.a_pretrain = self.trainer.apply_gradients(pre_grads_and_var)
        next_log_pis = a_dist_.log_prob(a_)
        with tf.control_dependencies(target_update):  # soft replacement happened at here
            min_next_q = tf.reduce_min([q1_,q2_],axis=0)
            q1_target = self.R + gamma * (1-self.terminal) * tf.stop_gradient(min_next_q - self.alpha * next_log_pis)    #ddpg
            q2_target = self.R + gamma * (1 - self.terminal) * tf.stop_gradient(min_next_q - self.alpha * next_log_pis)  # ddpg
            if self.approx_value:
                l_target = self.l_R + gamma * (1-self.terminal)*l_   # Lyapunov critic - self.alpha * next_log_pis
            else:
                l_target = self.l_R
            self.c1_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.c2_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.l_trainer = tf.train.AdamOptimizer(self.LR_C)
            self.td_error1 = tf.losses.mean_squared_error(labels=q1_target, predictions=self.q1)
            self.td_error2 = tf.losses.mean_squared_error(labels=q2_target, predictions=self.q2)
            self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
            c1_grads_and_var = self.c1_trainer.compute_gradients(self.td_error1, c1_params)
            c2_grads_and_var = self.c2_trainer.compute_gradients(self.td_error2, c2_params)
            l_grads_and_var = self.l_trainer.compute_gradients(self.l_error, l_params)
            c1_grads, c1_var = zip(*c1_grads_and_var)
            c2_grads, c2_var = zip(*c2_grads_and_var)
            l_grads, l_var = zip(*l_grads_and_var)
            # if self.max_grad_norm is not None:
            #     # Clip the gradients (normalize)
            #     c1_grads, _ = tf.clip_by_global_norm(c1_grads, self.max_grad_norm)
            #     c2_grads, _ = tf.clip_by_global_norm(c2_grads, self.max_grad_norm)
            #     l_grads, _ = tf.clip_by_global_norm(l_grads, self.max_grad_norm)
            c1_grads_and_var = list(zip(c1_grads, c1_var))
            c2_grads_and_var = list(zip(c2_grads, c2_var))
            l_grads_and_var = list(zip(l_grads, l_var))
            self.ctrain1 = self.c1_trainer.apply_gradients(c1_grads_and_var)
            self.ctrain2 = self.c2_trainer.apply_gradients(c2_grads_and_var)
            self.ltrain = self.l_trainer.apply_gradients(l_grads_and_var)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [ self.ctrain1, self.ctrain2, ]
        self.diag_names = ['labda', 'alpha', 'critic1_error','critic2_error','lyapunov_error', 'entropy', 'policy_loss']
        self.diagnotics = [self.labda, self.alpha, self.td_error1, self.td_error2, self.l_error, tf.reduce_mean(-self.log_pis)]

        if self.adaptive_alpha is True:
            self.opt.append(self.alpha_train)
        if self.use_lyapunov is True:
            self.opt.extend([self.ltrain, self.lambda_train])


    def choose_action(self, s, evaluation = False):
        if evaluation is True:
            return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]





    def learn(self, LR_A, LR_C, LR_L):
        if self.pointer>=self.memory_capacity:
            indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]  # state
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]  # action
        br = bt[:, -self.s_dim - 3: -self.s_dim - 2]  # reward
        blr = bt[:, -self.s_dim - 2: -self.s_dim -1]  # l_reward
        bterminal = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]  # next state

        feed_dict = {self.a_input: ba, self.S: bs, self.S_: bs_, self.R: br, self.l_R: blr, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L}

        if self.use_lyapunov is True:
            # 边缘的 s a s_ l_r
            if self.cons_pointer <= self.batch_size:
                opt_list = self.opt[:-1] + [self.a_pretrain]
                diagnotics = self.diagnotics + [self.a_preloss]
            else:
                if self.cons_pointer >= self.cons_memory_capacity:
                    indices = np.random.choice(self.cons_memory_capacity, size=self.batch_size)
                else:
                    indices = np.random.choice(self.cons_pointer, size=self.batch_size)

                bt = self.cons_memory[indices, :]
                cons_bs = bt[:, :self.s_dim]
                cons_ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
                cons_bs_ = bt[:, -self.s_dim:]
                cons_blr = bt[:, -self.s_dim - 2: -self.s_dim -1]

                feed_dict.update({self.cons_a_input: cons_ba, self.cons_S: cons_bs, self.cons_S_: cons_bs_,
                                  self.cons_l_R: cons_blr})
                opt_list = self.opt + [self.atrain]
                diagnotics = self.diagnotics + [ self.a_loss]
        else:
            opt_list = self.opt + [self.atrain]
            diagnotics = self.diagnotics + [ self.a_loss]


        self.sess.run(opt_list, feed_dict)
        diagnotics = self.sess.run(diagnotics, feed_dict)

        return diagnotics

    def store_transition(self, s, a, r, l_r, terminal, s_):
        transition = np.hstack((s, a, [r], [l_r], [terminal], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def store_edge_transition(self, s, a,r, l_r, terminal, s_):
        """把数据存入constraint buffer"""
        transition = np.hstack((s, a,[r], [l_r], [terminal], s_))
        index = self.cons_pointer % self.cons_memory_capacity  # replace the old memory with new memory
        self.cons_memory[index, :] = transition
        self.cons_pointer += 1

    #action 选择模块也是actor模块


    def _build_a(self, s, name='Actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)#原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_1, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)


            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=mu,
                    scale_diag=sigma),
            ))
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector)

            clipped_mu = squash_bijector.forward(mu)


        return clipped_a, clipped_mu, distribution



    #critic模块
    def _build_c(self, s, a, name ='Critic', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, 1, trainable=trainable)  # Q(s,a)


    def save_result(self, path):

        save_path = self.saver.save(self.sess, path+"/model.ckpt")
        print("Save to path: ", save_path)


def train(variant):
    env_name = variant['env_name']
    env = get_env_from_name(env_name)
    if variant['evaluate'] is True:
        evaluation_env = get_env_from_name(env_name)
    else:
        evaluation_env = None
    env_params = variant['env_params']
    judge_safety_func = get_safety_constraint_func(variant)

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']
    num_of_paths = variant['num_of_paths']

    alg_name = variant['algorithm_name']
    policy_build_fn = get_policy(alg_name)
    policy_params = variant['alg_params']
    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])

    logger.logkv('tau', policy_params['tau'])
    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = policy_build_fn(a_dim, s_dim, policy_params)
    logger.logkv('target_entropy', policy.target_entropy)
    # For analyse

    Render = env_params['eval_render']
    ewma_p = 0.95
    ewma_step = np.zeros((1, max_episodes + 1))
    ewma_reward = np.zeros((1, max_episodes + 1))

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False
    for i in range(max_episodes):

        ep_reward = 0
        l_r = 0

        current_path = {'rewards': [],
                        'l_rewards': [],
                        'violation':[],
                        }
        [current_path.update({key:[]}) for key in policy.diag_names]
        if global_step > max_global_steps:
            break

        s = env.reset()

        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            # Run in simulator
            s_, r, done, info = env.step(action)
            if training_started:
                global_step += 1
            l_r = info['l_rewards']
            if j == max_ep_steps - 1:
                done = True
            terminal = 1. if done else 0.

            violation_of_constraint = info['violation_of_constraint']
            # 储存s,a和s_next,reward用于DDPG的学习
            policy.store_transition(s, a, r, l_r, terminal, s_)

            # 如果状态接近边缘 就存储到边缘memory里
            # if policy.use_lyapunov is True and np.abs(s[0]) > env.cons_pos:  # or np.abs(s[2]) > env.theta_threshold_radians*0.8
            if policy.use_lyapunov is True and judge_safety_func(s_, r, done, info):  # or np.abs(s[2]) > env.theta_threshold_radians*0.8
                policy.store_edge_transition(s, a, r, l_r, terminal, s_)

            # Learn
            if policy.use_lyapunov is True:
                if policy.pointer > min_memory_size and global_step % steps_per_cycle == 0:
                    # Decay the action randomness
                    training_started = True
                    for _ in range(train_per_cycle):
                        train_diagnotic = policy.learn(lr_a_now, lr_c_now, lr_l_now)

            else:
                if policy.pointer > min_memory_size and global_step % steps_per_cycle == 0:
                    # Decay the action randomness
                    training_started = True
                    for _ in range(train_per_cycle):
                        train_diagnotic = policy.learn(lr_a_now, lr_c_now, lr_l_now)

            if training_started:
                current_path['rewards'].append(r)
                current_path['l_rewards'].append(l_r)
                current_path['violation'].append(violation_of_constraint)
                [current_path[key].append(value) for key,value in zip(policy.diag_names, train_diagnotic)]

            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:
                if evaluation_env is not None:
                    rollouts = get_evaluation_rollouts(policy, evaluation_env, num_of_paths, max_ep_steps,render=Render)
                    diagnotic = evaluate_rollouts(rollouts)
                    # [diagnotics[key].append(diagnotic[key]) for key in diagnotic.keys()]
                    print('training_step:', global_step, 'average eval reward:', diagnotic['return-average'],
                          'average eval lreward:', diagnotic['lreturn-average'],
                          'average eval violations:', diagnotic['violation-avg'],
                          'average length:', diagnotic['episode-length-avg'], )
                    logger.logkv('eval_eprewmean', diagnotic['return-average'])
                    logger.logkv('eval_eplrewmean', diagnotic['lreturn-average'])
                    logger.logkv('eval_eplenmean', diagnotic['episode-length-avg'])
                    logger.logkv('eval_violation_times', diagnotic['violation-avg'])
                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    # [training_diagnotics[key].append(training_diagnotic[key]) for key in training_diagnotic.keys()]\
                    logger.logkv('eprewmean', training_diagnotic['rewards'])
                    logger.logkv('eplrewmean', training_diagnotic['l_rewards'])
                    logger.logkv('eplenmean', training_diagnotic['len'])
                    logger.logkv('end_cost', training_diagnotic['end_cost'])
                    [logger.logkv(key, training_diagnotic[key]) for key in policy.diag_names]

                    logger.logkv('violation_times', training_diagnotic['violation'])
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    print('training_step:', global_step,
                          'average reward:', round(training_diagnotic['rewards'], 2),
                          'average lreward:', round(training_diagnotic['l_rewards'], 2),
                          'average violations:', training_diagnotic['violation'],
                          'end cost:', round(training_diagnotic['end_cost'],2),
                          'average length:', round(training_diagnotic['len'], 1),
                          'lyapunov error:', round(training_diagnotic['lyapunov_error'], 6),
                          'critic1 error:', round(training_diagnotic['critic1_error'], 6),
                          'critic2 error:', round(training_diagnotic['critic2_error'], 6),
                          'policy_loss:', round(training_diagnotic['policy_loss'], 6),
                          'alpha:', round(training_diagnotic['alpha'], 6),
                          'lambda:', round(training_diagnotic['labda'], 6),
                          'entropy:', round(training_diagnotic['entropy'], 6),)
                          # 'max_grad:', round(training_diagnotic['max_grad'], 6)
                logger.dumpkvs()
            # 状态更新
            s = s_
            ep_reward += r

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                ewma_step[0, i + 1] = ewma_p * ewma_step[0, i] + (1 - ewma_p) * j
                ewma_reward[0, i + 1] = ewma_p * ewma_reward[0, i] + (1 - ewma_p) * ep_reward
                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic

                break
    policy.save_result(log_path)
    print('Running time: ', time.time() - t1)
    return

