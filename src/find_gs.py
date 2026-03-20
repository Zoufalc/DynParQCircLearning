# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for a given Hamiltonian using the dynamic parameterized quantum circuit ansatz."""
import abc
import sys
from functools import partial

from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Tuple, List

import tensorcircuit as tc
import tensorcircuit.quantum as qu
import tensorcircuit.noisemodel as noisemodel
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
from src.utilities.result_saver import ResultSaver

def _make_jit_helpers(ansatz: Any) -> Tuple[Callable, Callable]:
    """
    Create JIT helpers for a class implementing `energy_from_params`:

    * costs_vmapped  - vectorised (batched) energy evaluation over params
    * cost_vvag      - vectorised value & gradient for parameter batches
    
    This function captures necessary data from ansatz without passing the entire
    object to JIT, preventing recompilation on every call.
    """
    energy_fn = ansatz.energy_from_params
    perform_noisy_simulations = getattr(ansatz, "perform_noisy_simulations", False)
    
    if perform_noisy_simulations:
        # For noisy simulations, we need to handle seeds
        def energy_with_seed(params, seed):
            return energy_fn(params, seed=seed)
        
        costs_vmapped = K.jit(K.vmap(energy_with_seed, vectorized_argnums=[0, 1]))
        cost_vvag = K.jit(K.vmap(K.value_and_grad(energy_with_seed, argnums=0),
                                 vectorized_argnums=[0, 1]))
    else:
        # For noiseless simulations
        costs_vmapped = K.jit(K.vmap(energy_fn, vectorized_argnums=0))
        cost_vvag = K.jit(K.vmap(K.value_and_grad(energy_fn, argnums=0),
                                 vectorized_argnums=0))
    
    return costs_vmapped, cost_vvag

HamiltonianTerm = Tuple[Tuple[Any, List[int]], float]

@dataclass(eq=False)
class VariationalAnsatz(abc.ABC):
    """
    Abstract base for all variational ansätze in this repo.

    Concrete subclasses need to implement abstract members:

    1. ``_circuit(self, params, *args)`` – builds the parameterised circuit.
    2. ``_hamiltonian_terms(self)`` – returns list of Hamiltonian terms.
    3. ``get_full_hamiltonian(self)`` – returns the sparse Hamiltonian operator.

    Everything else (optimisation loop, logging, initialisation, etc.) is
    provided here and works for *any* ansatz that respects those contracts.
    """

    # ---- generic optimisation hyper‑parameters (common to all ansätze) ----
    trials: int = 10                # number of independent random starts
    maxiter: int = 2001             # optimisation steps
    howoften_tosave: int = 10       # checkpoint frequency
    learning_rate: float = 1e-2     # Adam learning rate
    sparse: bool = True             # work with the sparse Hamiltonian representation
    
    # Noise simulation parameters
    perform_noisy_simulations: bool = False
    noise_rate: float = 1e-2
    number_of_shots: int = 1000

    # ---- fields that are filled automatically in __post_init__ ----
    nparams: int = field(init=False)          # derived from concrete class
    initparams: Any = field(init=False)       # shape (trials, nparams)
    fullham: Any = field(init=False, default=None)
    _costs_vmapped: Callable = field(init=False, repr=False)
    _cost_vvag: Callable = field(init=False, repr=False)

    @abc.abstractmethod
    def _circuit(self, params, *args):
        """Construct a circuit with the provided params."""
        ...

    @abc.abstractmethod
    def _hamiltonian_terms(self) -> List[HamiltonianTerm]:
        """Return list of Hamiltonian terms as (ops, coefficient) tuples."""
        ...

    @abc.abstractmethod
    def get_full_hamiltonian(self):
        """Build and return the full sparse Hamiltonian."""
        ...

    def __post_init__(self):
        """Initialize parameters and JIT helpers after dataclass initialization."""
        if self.sparse:
            print("Building full Hamiltonian")
            sys.stdout.flush()
            self.fullham = self.get_full_hamiltonian()
            print("Done")
        
        self.initparams = self._initialise_parameters()
        self._costs_vmapped, self._cost_vvag = _make_jit_helpers(self)

    def _initialise_parameters(self):
        """Initialize random parameters for all trials."""
        randint = np.random.randint(1e5)
        key = jax.random.PRNGKey(randint)
        return jax.random.uniform(key, shape=[self.trials, self.nparams],
                                 minval=0, maxval=jnp.pi)

    def energy_from_params(self, params, seed=None) -> Any:
        """Compute energy for given parameters."""
        qc = self._circuit(params, seed)
        
        if self.sparse:
            # Warn if noise is requested with sparse mode
            if self.perform_noisy_simulations:
                warnings.warn(
                    "Noisy simulations are not supported with sparse=True. "
                    "Set sparse=False to enable noisy simulations.",
                    UserWarning
                )
            return K.real(sparse_expectation(qc, self.fullham))
        else:
            # Non-sparse: compute expectation values term by term
            terms = self._hamiltonian_terms()
            energy = 0.0
            
            if self.perform_noisy_simulations:
                # Setup noise configuration
                noise_conf = noisemodel.NoiseConf()
                noise_conf.add_noise("depolarizing", [self.noise_rate*0.1],
                                    ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"])
                noise_conf.add_noise("depolarizing", [self.noise_rate],
                                    ["cnot", "cz", "swap", "iswap", "rxx", "ryy", "rzz"])
                
                nmc = self.number_of_shots
                if seed is not None:
                    K.set_random_state(seed)
                
                for ops, coeff in terms:
                    status = K.random_uniform(shape=[nmc], boundaries=(0.0, 1.0))
                    exp_val = tc.noisemodel.expectation_noisfy(
                        qc, *ops, noise_conf=noise_conf, nmc=nmc, status=status
                    )
                    energy += coeff * exp_val
            else:
                # Noiseless simulation
                for ops, coeff in terms:
                    exp_val = qc.expectation(*ops)
                    energy += coeff * exp_val
            
            return K.real(energy)

    def purity_from_params(self, params):
        """
        Calculate purity.
        Override this method in subclasses for system-specific purity calculations.
        """
        # This is a generic implementation that can be overridden
        if not hasattr(self, 'lattice'):
            raise NotImplementedError(
                "purity_from_params requires a 'lattice' attribute. "
                "Override this method in your subclass."
            )
        
        t = self.lattice
        n = t.num_qubits
        qc = self._circuit(params)
        
        s = qc.state()
        
        if qc._nqubits - n > n:
            cut = range(n)
        else:
            cut = range(n, qc._nqubits)
        
        rho = qu.reduced_density_matrix(s, cut=list(cut))
        return K.exp(-qu.renyi_entropy(rho, 2))

    def optimize(self, save_results: bool = False, track_purity: bool = False):
        """
        Run optimization to find ground state.
        
        Parameters:
        -----------
        save_results : bool
            If True, automatically save results after optimization completes
        track_purity : bool
            If True, track purity during optimization
            
        Returns:
        --------
        tuple : (final_energies, final_parameters, all_energies, all_purities)
        """
        params = jnp.array(self.initparams)
        self.allpurities = np.zeros((self.trials, 1 + (self.maxiter - 1) // self.howoften_tosave))
        self.allenergies = np.zeros((self.trials, 1 + (self.maxiter - 1) // self.howoften_tosave))
        counter = 0
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(params)

        # Initialize random seeds for noisy simulations
        if self.perform_noisy_simulations:
            base_seed = 42
            all_seeds = jnp.arange(self.maxiter * self.trials).reshape(self.maxiter, self.trials) + base_seed
        
        with tqdm(range(self.maxiter), miniters=self.howoften_tosave, mininterval=1) as pbar:
            for i in pbar:
                if self.perform_noisy_simulations:
                    seeds = all_seeds[i]
                    value, gradient = self._cost_vvag(params, seeds)
                else:
                    value, gradient = self._cost_vvag(params)
                
                updates, opt_state = optimizer.update(gradient, opt_state)
                params = optax.apply_updates(params, updates)
                
                if i % self.howoften_tosave == 0:
                    self.allenergies[:, counter] = value
                    if track_purity:
                        self.allpurities[:, counter] = purity_vec(self, params)
                    counter += 1
                    pbar.set_postfix_str(f"Current value: {str(jnp.min(value))}")

        # Optionally save results
        if save_results:
            self.save_results(value, params, self.allenergies, self.allpurities)

        return value, params, self.allenergies, self.allpurities

    def save_results(self, final_energies, final_parameters, all_energies, all_purities,
                     save_individual: bool = True):
        """
        Save optimization results to HDF5 files.
        
        This method saves:
        1. A record in the master file: results/<lattice_name>.h5
        2. An individual timestamped file in: tmp/<lattice_name>_TIMESTAMP.h5
        
        Parameters:
        -----------
        final_energies : array
            Final energy values from optimization
        final_parameters : array
            Final parameter values from optimization
        all_energies : array
            Energy trajectory during optimization
        all_purities : array
            Purity trajectory during optimization
        save_individual : bool
            If True, also save individual timestamped file in tmp/
            
        Returns:
        --------
        tuple : (master_file_path, individual_file_path or None)
        """
        from src.utilities.result_saver import ResultSaver
        
        # Initialize result saver
        saver = ResultSaver(self.lattice.name())
        
        # Prepare results dictionary
        results = {
            'final_energies': np.array(final_energies),
            'final_parameters': np.array(final_parameters),
            'all_energies': np.array(all_energies),
            'all_purities': np.array(all_purities),
            'min_energy': float(np.min(final_energies)),
            'mean_energy': float(np.mean(final_energies)),
            'std_energy': float(np.std(final_energies))
        }
        
        # Save using the result saver
        master_file, individual_file = saver.save_run(
            settings=self,
            results=results,
            save_individual=save_individual
        )
        
        print(f"Results saved to master file: {master_file}")
        if individual_file:
            print(f"Individual run saved to: {individual_file}")
        
        return master_file, individual_file

    def get_initial_costs(self, params=None):
        """Get initial cost values for all trials."""
        if params is None:
            params = self.initparams
        return self._costs_vmapped(params)

    def get_initial_costs_and_gradients(self, params=None):
        """Get initial costs and gradients for all trials."""
        if params is None:
            params = self.initparams
        return self._cost_vvag(params)

# JIT-compiled purity functions
def _purity_wrapper(ansatz, params):
    """Wrapper for JIT compilation of purity calculation."""
    return ansatz.purity_from_params(params)

purity = K.jit(_purity_wrapper, static_argnums=[0])
purity_vec = K.jit(K.vmap(_purity_wrapper, vectorized_argnums=[1]), static_argnums=[0])

# ToricCodeAnsatz has been moved to src/utilities/ansatz_classes.py
# Import it from there if needed
