# (C) Copyright IBM 2024.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functions to compute the gradients and QGTs needed to run a variational ground state search or Gibbs state
preparation."""

from tensorcircuit import experimental
from tensorcircuit.templates.measurements import operator_expectation
import jax.numpy as jnp

def grad_tc(ansatz, hamiltonian, backend): # n, l, trials=10
    def f(params):
        circ = ansatz(params)
        return operator_expectation(circ, hamiltonian)
    return backend.jit(backend.grad(f))


def qgt_tc(ansatz, backend): # n, l, trials=10
    def s(params):
        circ = ansatz(params)
        return circ.state()
    get_qgt_tc = backend.jit(experimental.qng(s, mode="fwd"))
    return get_qgt_tc

def state_grad(ansatz, backend):
    def f(params):
        dm = ansatz(params).densitymatrix()
        a,b = jnp.shape(dm)
        return jnp.reshape(dm, shape=a*b)
    return backend.jit(backend.jacfwd(f))


def grad_dm_tc(ansatz, hamiltonian, backend): # n, l, trials=10
    def f(params):
        dm = ansatz(params).densitymatrix()
        return backend.real(backend.trace(dm @ hamiltonian @ dm))
    return backend.jit(backend.grad(f))

def qgt_dm_tc(ansatz, backend):
    def s_(params, params_):
        circ = ansatz(params)
        dm = circ.densitymatrix()
        circ_ = ansatz(params_)
        dm_ = circ_.densitymatrix()
        return backend.real(backend.trace(backend.adjoint(dm) @ dm_))
    return backend.jit(backend.jacfwd(backend.grad(s_, argnums=0), argnums=1))



