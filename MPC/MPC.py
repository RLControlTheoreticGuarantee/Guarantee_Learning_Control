import numpy as np
import osqp
import scipy.sparse as sparse
import cvxpy as cp
import math
from gym import spaces

class MPC(object):
    def __init__(self, a_dim, s_dim, d_dim, variant):
        self.horizon = variant['horizon']
        theta_threshold_radians = 20 * 2 * math.pi / 360
        self.a_dim = a_dim
        self.s_dim = s_dim
        # self.s_bound = variant['s_bound']
        self.a_bound = variant['a_bound']
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 10
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50
        high = np.array([
            self.x_threshold,
            self.max_v,
            self.theta_threshold_radians,
            self.max_w])
        self.s_bound = spaces.Box(-high, high, dtype=np.float32)
        self.control_horizon = variant['control_horizon']
        length = 0.5
        tau = 0.005
        masscart = 1
        masspole = 0.1
        total_mass = (masspole + masscart)
        polemass_length = (masspole * length)
        g = 10
        H = np.array([
            [1, 0, 0, 0],
            [0, total_mass, 0, - polemass_length],
            [0, 0, 1, 0],
            [0, - polemass_length, 0, (2 * length) ** 2 * masspole / 3]
        ])

        Hinv = np.linalg.inv(H)

        self.A = Hinv @ np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, - polemass_length * g, 0]
        ])
        self.B = Hinv @ np.array([0, 1.0, 0, 0]).reshape((4, 1))
        self.A = np.eye(4) + tau * self.A
        self.B = tau * self.B

        # self.A = np.array([[1,	0.0200000000000000,	-0.000146262956164120,	-9.75295709398121e-07],
        #           [0,	1,	-0.0146184465178444,	-0.000146262956164120],
        #           [0,	0,	0.996782214976383,	0.0199785434944732],
        #           [0,	0,	-0.321605822193865,	0.996782214976383]])
        # self.B = np.array([[0.000195114814924033],
        #           [0.0195107679379903],
        #           [0.000292525910329313],
        #           [0.0292368928359034]])
        self.Q = np.diag([0, 0., 20 *(1/ theta_threshold_radians)**2, 0.])
        self.R = np.array([[0.1]])
        self.Q_N = np.diag([0, 0., 2000 *(1/ theta_threshold_radians)**2, 0.])


        # self.Q = np.diag([1.2, 0., 100., 0.])
        # self.R = np.array([[0.]])
        # self.Q_N = np.diag([10, 0., 1000., 0.])
        self.U_rate = np.array([[0.]])


        self.S = np.diag(1000000 * np.ones([s_dim + self.a_dim]))

        self.last_U = np.zeros([self.a_dim])
        self.X = cp.Variable((self.s_dim, self.horizon + 1))
        self.U = cp.Variable((self.a_dim, self.horizon + 1))
        self.W = cp.Variable((self.s_dim + self.a_dim, self.horizon + 1), nonneg=True)
        self.x_0 = cp.Parameter((self.s_dim))
        cost = 0
        constr = []

        for t in range(self.horizon):
            cost += cp.quad_form(self.X[:, t], self.Q) + cp.quad_form(self.U[:, t], self.R) \
                    + cp.quad_form(self.W[:, t + 1], self.S) + cp.quad_form(self.U[:, t+1]-self.U[:, t], self.U_rate)

            constr += [
                self.X[:, t + 1] - self.s_bound.high <= self.W[0:self.s_dim, t + 1],
                self.W[0:self.s_dim, t + 1] >= self.s_bound.low - self.X[:, t + 1],
                self.U[:, t] <= self.a_bound.high + self.W[self.s_dim:, t + 1],
                self.U[:, t] + self.W[self.s_dim:, t + 1] >= self.a_bound.low,
                # self.U[:, t] <= self.a_bound.high + self.W[self.s_dim:, t + 1],
                # self.U[:, t] + self.W[self.s_dim:, t + 1] >= self.a_bound.low,
            ]

            constr += [self.X[:, t + 1] == self.A@ self.X[:, t] + self.B @ self.U[:, t]]

        cost += cp.quad_form(self.X[:, -1], self.Q_N)
        # sums problem objectives and concatenates constraints.
        # cost += cp.quad_form(self.X[:, t+1], self.Q_N)
        constr += [self.X[:, 0] == self.x_0]
        self.problem = cp.Problem(cp.Minimize(cost), constr)
        self.control_pointer = 0
        # self.K = [13.7657004354315,	27.2115988456560,	60.7761993549902,	17.5879627975564]

    def choose_action(self, x_0, arg):
        if self.control_pointer == 0:
            self.x_0.value = x_0
            self.problem.solve(solver=cp.OSQP, warm_start=True)
        pointer = self.control_pointer
        prediction_error = np.linalg.norm(self.X[:, self.control_pointer].value - x_0)
        a = self.U[:, self.control_pointer].value
        self.control_pointer += 1
        if self.control_pointer == self.control_horizon:
            self.control_pointer = 0


        # a = self.K @ x_0
        return a


    def restore(self, log_path):

        return True

