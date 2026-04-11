# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for a given Hamiltonian using the dynamic parameterized quantum circuit ansatz."""
import abc
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Tuple, List

import numpy as np
import jax
from jax import numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import tensorcircuit as tc
import tensorcircuit.quantum as qu
from tensorcircuit.templates.measurements import sparse_expectation

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
import optax
import warnings
from tqdm import tqdm

def opt_reconf(inputs, output, size, **kws):
    tree = optr.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(
        progbar=True, num_trees=4, num_restarts=20, subtree_weight_what=("size",),
        parallel=False
    )
    return tree_r.get_path()
K = tc.set_backend("jax")
tc.set_contractor("custom", optimizer=optr, preprocessing=True)

warnings.filterwarnings(
    "ignore", message=".*The inputs or output of this tree are not ordered.*"
)

HamiltonianTerm = Tuple[Tuple[Any, List[int]], float]

def _make_jit_helpers(ansatz):
    """
    Create JIT helpers for a class implementing `energy_from_params`:

    * costs_vmapped  - vectorised (batched) energy evaluation over params
    * cost_vvag      - vectorised value & gradient for parameter batches
    
    This function captures necessary data from ansatz without passing the entire
    object to JIT, preventing recompilation on every call.
    """

    energy_fn = ansatz.energy_from_params

    if ansatz.perform_noisy_simulations:
        def wrapped(params, key):
            return energy_fn(params, key)

        costs_vmapped = K.jit(
            K.vmap(wrapped, vectorized_argnums=[0, 1]),
            static_argnums=(),
        )

        cost_vvag = K.jit(
            K.vmap(K.value_and_grad(wrapped), vectorized_argnums=[0, 1]),
            static_argnums=(),
        )
    else:
        costs_vmapped = K.jit(
            K.vmap(energy_fn, vectorized_argnums=[0])
        )

        cost_vvag = K.jit(
            K.vmap(K.value_and_grad(energy_fn), vectorized_argnums=[0])
        )

    return costs_vmapped, cost_vvag

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
    max_trials_in_batch: Optional[int] = None  # max trials to vmap over at once (None = all trials)
    
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
        key = jax.random.PRNGKey(np.random.randint(1e6))
        return jax.random.uniform(
            key,
            shape=(self.trials, self.nparams),
            minval=0.0,
            maxval=jnp.pi,
        )

    def energy_from_params(self, params, key=None):
        qc = self._circuit(params)

        if self.sparse:
            return K.real(sparse_expectation(qc, self.fullham))

        terms = self._hamiltonian_terms()
        energy = 0.0

        if not self.perform_noisy_simulations:
            for ops, coeff in terms:
                energy = energy + coeff * qc.expectation(*ops)
            return K.real(energy)

        # -------- noisy simulation --------

        nmc = self.number_of_shots
        noise_conf = tc.noisemodel.NoiseConf()
        noise_conf.add_noise(
            "depolarizing",
            [self.noise_rate * 0.1],
            ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"],
        )
        noise_conf.add_noise(
            "depolarizing",
            [self.noise_rate],
            ["cnot", "cz", "swap", "iswap", "rxx", "ryy", "rzz"],
        )

        assert key is not None, "PRNGKey required for noisy simulations"

        # Pre-split keys for all terms to avoid splitting inside vmap
        num_terms = len(terms)
        keys = jax.random.split(key, num_terms)
        
        for idx, (ops, coeff) in enumerate(terms):
            status = jax.random.uniform(keys[idx], shape=(nmc,))
            exp_val = tc.noisemodel.expectation_noisfy(
                qc,
                *ops,
                noise_conf=noise_conf,
                nmc=nmc,
                status=status,
            )
            energy = energy + coeff * exp_val

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
        
        batch_size = self.max_trials_in_batch or self.trials
        n_batches = (self.trials + batch_size - 1) // batch_size

        master_key = jax.random.PRNGKey(42)
        # Create keys for each iteration and trial
        # jax.random.split returns shape (n, 2), so we reshape to (maxiter, trials, 2)
        all_keys = jax.random.split(master_key, self.maxiter * self.trials)
        key_matrix = all_keys.reshape(self.maxiter, self.trials, 2)

        for batch_idx in range(n_batches):
            start_idx, end_idx = batch_idx * batch_size, min((batch_idx + 1) * batch_size, self.trials)
            batch_params = params[start_idx:end_idx]
            optimizer = optax.adam(self.learning_rate)
            opt_state = optimizer.init(batch_params)

            counter = 0
            with tqdm(range(self.maxiter)) as pbar:
                for i in pbar:
                    if self.perform_noisy_simulations:
                        keys = key_matrix[i, start_idx:end_idx, :]
                        values, gradients = self._cost_vvag(batch_params, keys)
                    else:
                        values, gradients = self._cost_vvag(batch_params)

                    updates, opt_state = optimizer.update(gradients, opt_state)
                    batch_params = optax.apply_updates(batch_params, updates)

                    if i % self.howoften_tosave == 0:
                        self.allenergies[start_idx:end_idx, counter] = values
                        if track_purity:
                            self.allpurities[start_idx:end_idx, counter] = purity_vec(self, batch_params)
                        counter += 1

                    pbar.set_postfix_str(f"min(E)={float(jnp.min(values)):.6f}")

            params = params.at[start_idx:end_idx].set(batch_params)

        # Evaluate final energies in batches to avoid OOM errors
        if self.max_trials_in_batch is not None and self.trials > self.max_trials_in_batch:
            # Batch evaluation to prevent memory issues
            value = jnp.zeros(self.trials)
            for batch_start in range(0, self.trials, self.max_trials_in_batch):
                batch_end = min(batch_start + self.max_trials_in_batch, self.trials)
                batch_params = params[batch_start:batch_end]
                
                if self.perform_noisy_simulations:
                    batch_keys = key_matrix[-1, batch_start:batch_end, :]
                    batch_value = self._costs_vmapped(batch_params, batch_keys)
                else:
                    batch_value = self._costs_vmapped(batch_params)
                
                value = value.at[batch_start:batch_end].set(batch_value)
        else:
            # Original single evaluation for small trial counts
            if self.perform_noisy_simulations:
                final_keys = key_matrix[-1, :, :]
                value = self._costs_vmapped(params, final_keys)
            else:
                value = self._costs_vmapped(params)
        
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
