import tensorflow as tf
import functools
import numpy as np
from common.tf_util import get_session, save_variables, load_variables
from common.tf_util import initialize

try:
    from common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from common.mpi_util import sync_from_root
except ImportError:
    MPI = None



class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model

    train():
    - Make the training part (feedforward and retropropagation of gradients)

    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef,lf_coef, max_grad_norm, ALPHA3=0.01,init_labda=1., microbatch_size=None,
                 use_adaptive_alpha3 = False, approximate_value_function = False):
        self.sess = sess = get_session()
        self.approximate_value_function = approximate_value_function
        with tf.variable_scope('ppo2_lyapunov_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        # 这两个R都是带衰减的R
        self.R = R = tf.placeholder(tf.float32, [None])
        self.R_l = R_l = tf.placeholder(tf.float32, [None])
        self.v_l = v_l = tf.placeholder(tf.float32, [None])
        log_labda = tf.get_variable('ppo2_lyapunov_model/Labda', None, tf.float32, initializer=tf.log(init_labda))
        self.labda = tf.exp(log_labda)
        self.ALPHA3 = tf.placeholder(tf.float32, None, 'ALPHA3')
        self.use_adaptive_alpha3 = use_adaptive_alpha3
        if use_adaptive_alpha3:
            self.alpha3 = 0.000000001
        else:
            self.alpha3 = ALPHA3
        # self.log_labda = tf.placeholder(tf.float32, None, 'Labda')
        # self.labda = tf.constant(10.)
        # self.Lam=10.

        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        self.OLDLPRED = OLDLPRED = tf.placeholder(tf.float32, [None])
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Get the predicted value
        lpred = train_model.lf
        lpredclipped = OLDLPRED + tf.clip_by_value(train_model.lf - OLDLPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        lf_losses1 = tf.square(lpred - v_l)
        # Clipped value
        lf_losses2 = tf.square(lpredclipped - v_l)

        lf_loss = .5 * tf.reduce_mean(tf.maximum(lf_losses1, lf_losses2))



        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)


        # Defining Lyapunov loss

        lpred=train_model.lf
        lpred_=train_model.lf_
        # self.l_lambda = tf.reduce_mean(ratio *  tf.stop_gradient(lpred_) - tf.stop_gradient(lpred))
        self.l_lambda = tf.reduce_mean(ratio * tf.stop_gradient(lpred_) - tf.stop_gradient(lpred)+self.ALPHA3*self.R_l)



        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))+ self.l_lambda*tf.stop_gradient(self.labda) - \
                  tf.stop_gradient(self.l_lambda) * log_labda
        # pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)+ self.l_lambda * self.labda)
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + lf_loss * lf_coef

        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_lyapunov_model')
        # 2. Build our trainer
        if MPI is not None:
            self.trainer = MpiAdamOptimizer(MPI.COMM_WORLD, learning_rate=LR, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'lyapunov_loss','policy_entropy', 'approxkl', 'clipfrac', 'lyapunov_lambda']
        self.stats_list = [pg_loss, vf_loss,lf_loss,entropy, approxkl, clipfrac, self.labda]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.eval_step = act_model.eval_step
        self.value = act_model.value
        self.l_value = act_model.l_value
        self.l_value_ = act_model.l_value_
        self.initial_state = act_model.initial_state

        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, obs_,returns,l_returns, masks, actions, values,l_values,mb_l_rewards, neglogpacs,states=None):
        # print(lr, cliprange, obs, obs_,returns,l_returns, masks, actions, values,l_values, neglogpacs)
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')

        advs = returns - values
        l_advs= l_returns - l_values
        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        l_advs = (l_advs - l_advs.mean()) / (l_advs.std() + 1e-8)
        # labda_map = {
        #     self.train_model.X : obs,
        #     self.train_model.X_: obs_,
        #     self.A : actions,
        #     self.ADV : advs,
        #     self.R : returns,
        #     self.R_l: mb_l_rewards,
        #     self.LR : lr,
        #     self.CLIPRANGE : cliprange,
        #     self.OLDNEGLOGPAC : neglogpacs,
        #     self.OLDLPRED: l_values,
        #     self.OLDVPRED: values,
        # }
        # tol=0.001
        # l_q = self.sess.run(self.l_lambda, labda_map)
        #
        # if l_q > tol:
        #     if self.Lam == 0:
        #         self.Lam= 1e-8
        #     self.Lam = min(self.Lam * 2, 1e2)
        # elif l_q < -tol:
        #     self.Lam = self.Lam / 2
        # print(self.Lam)

        td_map = {
            self.train_model.X : obs,
            self.train_model.X_: obs_,
            self.A : actions,
            self.ADV : advs,
            # self.labda:self.Lam,
            self.R : returns,
            self.R_l: mb_l_rewards,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values,
            self.OLDLPRED: l_values,
            self.ALPHA3: self.alpha3
        }
        if self.approximate_value_function is True:
            td_map.update({self.v_l: l_returns})
        else:
            td_map.update({self.v_l: mb_l_rewards})
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(self.stats_list + [self._train_op],td_map)[:-1]

