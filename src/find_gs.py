# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for a given Hamiltonian using the dynamic parameterized quantum circuit ansatz."""
import sys
from functools import partial

from scipy.optimize import minimize
from dataclasses import dataclass

import tensorcircuit as tc
import tensorcircuit.quantum as qu
from tensorcircuit.templates.measurements import operator_expectation, sparse_expectation
from jax import config
from tqdm import tqdm
config.update("jax_enable_x64", True)
from jax import numpy as jnp
import cotengra as ctg
optr = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="combo",
    # max_time=120,
    max_repeats=200,
    progbar=True,
    directory=True,
)
import warnings
warnings.filterwarnings("ignore", message=".*The inputs or output of this tree are not ordered.*")

def opt_reconf(inputs, output, size, **kws):
    tree = optr.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(
        progbar=True, num_trees=4, num_restarts=20, subtree_weight_what=("size",),
        parallel=False
    )
    return tree_r.get_path()
K = tc.set_backend("jax")
tc.set_contractor("custom", optimizer=optr, preprocessing=True)

import optax

from src.utilities.generate_ansatz import *

def energy_from_params(self,params):
    qc = construct_dyn_circuit_toriccodelattice(params,self.Lx,self.Ly,self.nlayers,self.howoften_toreset)
    return sparse_expectation(qc,self.fullham)

cost_grad = K.jit(K.value_and_grad(energy_from_params,argnums=[1]), static_argnums=[0])
cost_vvag = K.jit(K.vmap(cost_grad,
                        vectorized_argnums=[1]
                        ),
                    static_argnums=[0]
                    )

def purity_from_params(self,params):
    t = ToricCode(self.Lx,self.Ly)
    n = t.num_qubits
    qc = construct_dyn_circuit_toriccodelattice(params,self.Lx,self.Ly,self.nlayers,self.howoften_toreset)
    s = qc.state()
    if qc._nqubits - n > n:
        cut = range(n)
    else:
        cut = range(n,qc._nqubits)

    rho = qu.reduced_density_matrix(s, cut=list(cut))
    return K.exp(-qu.renyi_entropy(rho,2))

purity = K.jit(purity_from_params,static_argnums=[0])
purity_vec = K.jit(K.vmap(purity,vectorized_argnums=[1]),static_argnums=[0])

@dataclass
class ToricCodeAnsatz:
    """
    A class that manages all relevant settings, including optimization options.
    """
    Lx: int = 2
    Ly: int = 2
    nlayers: int = 2
    howoften_toreset: int = 1
    h : float = 0.0
    trials: int = 10
    maxiter: int = 2001
    howoften_tosave: int = 10
    learning_rate : float = 1e-2

    def __post_init__(self):
        self.tc = ToricCode(self.Lx,self.Ly)
        self.nplaquettes = (self.Lx-1)*(self.Ly-1)
        self.nmeasurements = self.nplaquettes * (self.nlayers//self.howoften_toreset)
        self.nparams = self.nplaquettes * 4 * 9 *self.nlayers + 3*self.tc.num_qubits
        self.nancillas = self.nplaquettes + self.nplaquettes * (self.nlayers//self.howoften_toreset)
        print(self.__dict__)
        print("Building full Hamiltonian")
        sys.stdout.flush()
        self.fullham = self.get_full_hamiltonian()
        print("Done")
        self.initparams = self.initialise_parameters()

    def __hash__(self):
        return hash((self.Lx, self.Ly, self.nlayers, self.howoften_toreset, self.h, self.trials, self.maxiter, self.howoften_tosave, self.learning_rate))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_full_hamiltonian(self):
        hamiltonian = perturbed_hamiltonian(self.Lx,self.Ly,self.h,self.nancillas)
        strings, weights = self.tc.hamiltonian_tc(1-self.h,self.nancillas)
        perturbed_strings, perturbed_weights = self.tc.hamiltonian_tc_perturbation(self.h,self.nancillas)

        strings.extend(perturbed_strings)
        weights = np.concatenate((weights,perturbed_weights))
        self.fullham = qu.PauliStringSum2COO(strings, weights)
        return self.fullham

    def initialise_parameters(self):
        randint = np.random.randint(1e5)
        key = jax.random.PRNGKey(randint)
        param = jax.random.uniform(key,
        shape=[self.trials,self.nparams],
        minval=0,maxval=jnp.pi)
        return param

    def optimize(self):
        params = jnp.array(self.initparams)
        self.allpurities = np.zeros((self.trials,1+ self.maxiter//self.howoften_tosave))
        self.allenergies = np.zeros((self.trials,1+ self.maxiter//self.howoften_tosave))
        counter = 0
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(params)

        with tqdm(range(self.maxiter), miniters=self.howoften_tosave,mininterval=1) as pbar:
            for i in pbar:
                value, gradient = cost_vvag(self,params)
                updates, opt_state = optimizer.update(gradient[0], opt_state)
                params = optax.apply_updates(params, updates)
                if i % self.howoften_tosave == 0:
                    self.allenergies[:,counter] = value
                    self.allpurities[:,counter] = purity_vec(self,params)
                    counter += 1
                    pbar.set_postfix_str(f"Current value: {str(jnp.min(value))}")


        return value, params, self.allenergies, self.allpurities