# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Variational quantum time evolution algorithm."""

import warnings
from typing import Callable, Optional, Iterable

import logging

import scipy as sp
import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator

logger = logging.getLogger(__name__)


class VarQTE:
    """Variational quantum time evolution class."""

    def __init__(
        self,
        ansatz,
        initial_parameters,
        gradient,
        qgt,
        backend,
        sle_solver: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        store_gradients: bool = False,
        dm: bool = False,

    ) -> None:
        """
        Args:
            ansatz: The parameterized circuit used for the variational time evolution.
            initial_parameters: The initial parameters for the circuit.
            estimator: The estimator to run expectation values.
            gradient: The gradient object for the evolution gradient.
            qgt: The QGT evaluation object.
            sle_solver: The LSE solver to solve for the parameter derivative. Defaults to
                ridge regularization.
            store_gradients: If ``True``, store the QGTs and gradients.
            dm: Set True if VarQTE with dm instead of unitary ansatz
        """
        self.ansatz = ansatz
        self.initial_parameters = initial_parameters
        self.gradient = gradient
        self.qgt = qgt
        self.store_gradients = store_gradients
        self.dm = dm
        self.backend = backend

        if sle_solver is None:  # L-curve regularization per default

            def stable_solver(G, b, tol=1e-3, norm_threshold=1e4):
                if not np.allclose(G, G.T):
                    warnings.warn("G is not symmetric!")
                    G = (G + G.T) / 2

                v, U = np.linalg.eigh(G)

                if not np.allclose(G, U @ np.diag(v) @ U.T):
                    warnings.warn("decomp failed!")

                b_U = U.T @ b

                idx = np.where(v < tol)[0]

                x_U = [0 if i in idx else b_U[i] / v[i] for i, _ in enumerate(b)]
                x = U @ x_U

                if np.linalg.norm(x) / x.size > norm_threshold:
                    warnings.warn(f"Norm of update step exceeded threshold: {np.linalg.norm(x)}")
                return x.real

        self.sle_solver = stable_solver


    def infidelity(self,
                   hamiltonian: BaseOperator,
                   params: Iterable,
                   inv_temp: float) -> float:
        """
        Return infidelity
        :param hamiltonian: Hamiltonian underlying the Gibbs state
        :param params: Values of ansatz parameters.
        :param inv_temp: Inverse temperature corresponding to Gibbs state.
        :return: Infidelity between target state and currently prepared state.
        """

        state = self.ansatz(params)
        prep_dens = state.densitymatrix()
        hamiltonian = np.array(hamiltonian)*inv_temp*(-1)
        rho = self.backend.expm(hamiltonian) #τ = 1/2 (kBT)
        target_dens  = rho / self.backend.trace(rho)
        fid = self.backend.trace(self.backend.sqrtmh(prep_dens @ target_dens, psd=True)) ** 2
        fid = self.backend.real(fid)
        infid = 1 - fid
        return self.backend.real(infid)  # tc_inf

    def res_helper(self, hamiltonian, params):
        """
        Evaluate Additional residual factor.
        :param hamiltonian: Hamiltonian for the variational quantum time evolution.
        :param params: Current parameter values.
        :return: Additional residual factor.
        """
        state = self.ansatz(params)
        prep_dens = state.densitymatrix()
        energy = self.backend.trace(np.matmul(hamiltonian, prep_dens))
        var_temp = np.matmul(hamiltonian, prep_dens) + np.matmul(prep_dens, hamiltonian) - 2 * prep_dens * energy
        res_temp = self.backend.trace(np.matmul(var_temp, np.conj(np.transpose(var_temp))))
        return res_temp

    def residual(self, A, b, theta_dot, res_temp) -> float:
        """
        Evaluate residual error of variational principle.
        :param A: Quantum geometric tensor.
        :param b: Right-hand side.
        :param theta_dot: Parameter update.
        :param res_temp: Additional residual factor.
        :return: Residual of a single variational quantum time evolution step.
        """
        theta_dot = np.reshape(theta_dot, (np.shape(theta_dot)[0], 1))
        res = np.matmul(np.transpose(theta_dot), np.matmul(A, theta_dot))[0][0]
        print('QGT part ', res)
        res -= 2*self.backend.real(np.matmul(b, theta_dot))[0]
        print('2*Ytimes dot ', 2*np.matmul(b, theta_dot)[0])
        res += res_temp
        return np.real(res)

    def evolve(
        self,
        hamiltonian: BaseOperator,
        final_time: float,
        stepsize: float,
        res_ = False) -> Iterable:

        """
        Evolve state according to imaginary time evolution
        :param hamiltonian: Hamiltonian underlying the Gibbs state.
        :param final_time: Target time corresponding to inverse temperature for Gibbs state.
        :param stepsize: Step size used for the ODE discretization of the evolution w.r.t. the system temperature.
        :param res_: If True use residual minimization to improve the step parameters
        :return: Final values for ansatz parameters which give the Gibbs state approximation.
        """

        times = [0]
        x = [self.initial_parameters]
        qgts = []
        bs = []

        while times[-1] < final_time:
            if self.dm:
                b_ = np.real(self.gradient(x[-1][0]))
            else:
                b_ = np.real(self.gradient(x[-1][0])) * -0.5
            if np.any(np.isnan(b_)):
                b_ = np.nan_to_num(b_)
                raise Warning('NAN values set to 0 in the gradient')

            if self.dm:
                qgt_ = np.real(self.qgt(x[-1][0], x[-1][0]))
                if np.any(np.isnan(qgt_)):
                    qgt_ = np.nan_to_num(qgt_)
                    raise Warning('NAN values set to 0 in the QGT')

            if self.store_gradients:
                qgts.append(qgt_)
                bs.append(b_)


            step = self.sle_solver(qgt_.real, b_.real)

            var_temp = self.res_helper(hamiltonian, params=x[-1][0])
            if res_ == True:
                res_fun = lambda y: self.residual(qgt_, b_, y, var_temp)
                result = sp.optimize.minimize(res_fun, x0=step, method='COBYLA',
                                              options={'maxfun': 100})
                step = result.x

            x.append(x[-1] + stepsize * step)
            times.append(times[-1] + stepsize)


        return x #parameters
