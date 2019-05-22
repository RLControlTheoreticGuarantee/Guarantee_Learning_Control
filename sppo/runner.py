import numpy as np
from common.runners import AbstractEnvRunner
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, safety_gamma, lam, evaluation_frequency = 1000, n_of_paths=10):
        super().__init__(env=env, model=model, nsteps=nsteps)

        self.evaluation_frequency = evaluation_frequency
        self.n_of_paths = n_of_paths
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.safety_gamma = safety_gamma

    def run(self,RENDER=False):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs,mb_obs_, mb_rewards,mb_l_rewards, mb_actions, mb_values,mb_l_values,mb_l_values_,mb_dones, mb_neglogpacs = [],[],[],[],[],[],[],[],[],[]
        mb_states = self.states#model.initial_state
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # 这个step来自于policies其实
            # 这两个value都属于v value
            # 这一步用来得到当前state下选择的action 以及当前state下的v 和 lv

            actions, values, l_values,self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_l_values.append(l_values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            # env里需要有lyapunov 的reward 给出
            self.obs[:],rewards, self.dones, infos = self.env.step(actions)
            if RENDER==True:
                self.env.render()
            mb_obs_.append(self.obs.copy())

            l_rewards = [info['l_rewards'] for info in infos]
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            mb_l_rewards.append(l_rewards)


        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_ = np.asarray(mb_obs_, dtype=self.obs_.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_l_rewards = np.asarray(mb_l_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_l_values = np.asarray(mb_l_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)
        last_l_values = self.model.l_value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        # discount/bootstrap off value fn
        mb_l_returns = np.zeros_like(mb_l_rewards)
        mb_l_advs = np.zeros_like(mb_l_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_l_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_l_values[t + 1]
            delta = mb_l_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_l_values[t]
            mb_l_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_l_returns = mb_l_advs + mb_l_values

        # print(np.shape(mb_returns),np.shape(mb_l_returns),np.shape(mb_values),np.shape(mb_l_values),np.shape(mb_rewards),np.shape(mb_l_rewards))
        return (*map(sf01, (
        mb_obs, mb_obs_, mb_returns, mb_l_returns, mb_dones, mb_actions, mb_values, mb_l_values, mb_l_rewards,mb_neglogpacs)),
            mb_states, epinfos)
    def eval_run(self):
        evalinfos = []
        for _ in range(self.n_of_paths):
            ep_ends = False
            while True:

                actions, values, l_values, self.states, neglogpacs = self.model.eval_step(self.obs, S=self.states, M=self.dones)
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        evalinfos.append(maybeepinfo)
                        ep_ends = True
                if ep_ends:
                    break

        eval_rewards = [path['r'] for path in evalinfos]
        eval_lrewards = [path['lr'] for path in evalinfos]
        eval_length = [path['l'] for path in evalinfos]

        eval_info = {'mean_rewards':round(np.mean(eval_rewards),6),
                     'max_rewards': round(np.max(eval_rewards), 6),
                     'min_rewards': round(np.min(eval_rewards), 6),
                     'mean_lrewards': round(np.mean(eval_lrewards), 6),
                     'max_lrewards': round(np.max(eval_lrewards), 6),
                     'min_lrewards': round(np.min(eval_lrewards), 6),
                     'mean_length': round(np.mean(eval_length), 6),
                     }
        return eval_info



# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


