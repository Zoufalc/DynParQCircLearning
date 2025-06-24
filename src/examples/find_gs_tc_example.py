# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for Toric Code Hamiltonian using the dynamic parameterized quantum circuit ansatz."""

from src.utilities.generate_toric_code_hamiltonian import *
from src.utilities.generate_ansatz import *
from src.find_gs import *

# Define system parameters
Lx = 3
Ly = 2
nlayers = 2
howoften_toreset = 1
howoften_tosave = 10
tc_ = ToricCode(Lx,Ly)
h = 0.0
trials = 4
maxiter = 401
learning_rate = 1e-2

if __name__ == "__main__":
    ansatz = ToricCodeAnsatz(Lx, Ly,nlayers,howoften_toreset,h,trials,maxiter,howoften_tosave)
    final_energies, final_parameters, all_energy_values, all_purity_values = ansatz.optimize()
    print('final values ', final_energies)
    print('final purities ', all_purity_values[:,-1])