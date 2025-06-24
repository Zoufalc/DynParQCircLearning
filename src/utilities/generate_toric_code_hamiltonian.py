# This code is associated to the quantum optimization benchmarking effort
#
# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Toric Code Hamiltonian Generation."""

from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, partial_trace, entropy
import numpy as np
import scipy

#TODO Typehints and comments

I = Pauli('I')
Z = Pauli('Z')
X = Pauli('X')
Y = Pauli('Y')

def sparse_to_paulistring(indices, qubits, pauli='X'):
    string = ''
    for i in range(qubits):
        if i in indices:
            string = pauli + string # Needed because of little endian convention in Qiskit
        else:
            string = 'I' + string
    return string

def array_to_tc_structure(arrays,nqubits,pauli='X'):
    structure = []
    for ar in arrays:
        str = np.zeros(nqubits)
        for idx in ar:
            if pauli == 'X':
                str[idx] = 1
            if pauli == 'Y':
                str[idx] = 2
            if pauli == 'Z':
                str[idx] = 3
        structure.append(str)
        
    return list(structure)
            
def state_to_qiskit(original_state):
    return Statevector(original_state)

def perturbed_hamiltonian(Lx,Ly,h=0,nancillas=0,direction='Z'):
    tc = ToricCode(Lx,Ly)
    h0 = tc.hamiltonian(1-h,nancillas)
    list = [("Z", [i], h) for i in range(tc.num_qubits)]
    if direction=='XZ' or direction == 'ZX':
        list.extend([("X", [i], h) for i in range(tc.num_qubits)])
    h0 -= SparsePauliOp.from_sparse_list(list, num_qubits=tc.num_qubits + nancillas)
    return h0


class ToricCode:
    def __init__(self, Lx, Ly):
        self.Lx = Lx  # Lattice size in x direction
        self.Ly = Ly  # Lattice size in y direction
        self.num_qubits = 2 * Lx * Ly  - Lx - Ly # Each site has 2 qubits (one for each edge)

    def qubit_index(self, x, y, direction):
        """
        Maps a pair of integers (x, y) and a direction (0 for horizontal, 1 for vertical) to a qubit index.
        If direction is 2, then just gives the corner vertex (used when adding ancillas to toric code)
        """
        
        if x==self.Lx-1 and direction==0:
            return None
        if y==self.Ly-1 and direction==1:
            return None
        if x < 0 or x >= self.Lx:
            return None
        if y < 0 or y >= self.Ly:
            return None
        if direction == 0:
            return (y * (self.Lx - 1) + x)
        if direction == 1:
            return (y * (self.Lx) + x + self.Ly*(self.Lx - 1))
        if direction == 2:
            return (y * (self.Lx-1) + x + 2*self.Ly*(self.Lx) - self.Lx - self.Ly)
    
    def claws(self, x, y):
        """
        Given a qubit "address", generates the pairs of 2-qubits we'll use in the FLDC.
        """
        candidate = [(self.qubit_index(x, y+1, 0), self.qubit_index(x, y, 1)),
        (self.qubit_index(x, y+1, 0), self.qubit_index(x, y, 0)),
        (self.qubit_index(x, y+1, 0), self.qubit_index(x+1, y, 1))
        ]
        return list(filter(lambda x: not None in x, candidate))

    def claws_measurements(self, x, y):
        """
        Given a qubit "address", generates the pairs of 2-qubits we'll use in the FLDC.
        """
        candidate = [(self.qubit_index(x, y+1, 0), self.qubit_index(x, y, 1)), # First
        (self.qubit_index(x, y+1, 0), self.qubit_index(x+1, y, 1)),
        (self.qubit_index(x+1, y, 1), self.qubit_index(x, y, 0)), # These first three define a U-shaped claw
        (self.qubit_index(x, y, 2), self.qubit_index(x, y, 0)), # Ancilla
        # (self.qubit_index(x, y, 2), self.qubit_index(x+1, y, 1)), # Ancilla
        ]
        return list(filter(lambda x: not None in x, candidate))

    def all_claws(self):
        return sum([self.claws(x,y) for y in range(self.Ly-1) for x in range(self.Lx-1)],[])

    def all_claws_measurements(self):
        return sum([self.claws_measurements(x, y) for y in range(self.Ly - 1) for x in range(self.Lx - 1)], [])

    def star_operator(self, x, y, nancillas=0, tc=0):
        """
        Returns the qubit indices involved in the star operator at vertex (x, y).
        The star operator is a tensor product of Pauli Z operators on the qubits
        around the vertex (x, y).
        """
        indices = [
            self.qubit_index(x, y, 0),                # Horizontal qubit on (x, y)
            self.qubit_index((x - 1), y, 0), # Horizontal qubit on (x-1, y)
            self.qubit_index(x, (y - 1), 1), # Vertical qubit on (x, y-1)
            self.qubit_index(x, y, 1)                 # Vertical qubit on (x, y)
        ]
        zops = list(filter(lambda x: x != None, indices))
        zstring = sparse_to_paulistring(zops, self.num_qubits + nancillas, 'Z')
        if tc == 0:
            return zstring
        else:
            return zops

    def plaquette_operator(self, x, y, nancillas=0, tc=0):
        """
        Returns the qubit indices involved in the plaquette operator at plaquette (x, y).
        The plaquette operator is a tensor product of Pauli X operators on the qubits
        around the plaquette (x, y).
        """
        indices = [
            self.qubit_index(x, y, 0),               # Horizontal qubit on (x, y)
            self.qubit_index((x + 1), y, 1), # Vertical qubit on (x+1, y)
            self.qubit_index(x, (y + 1), 0), # Horizontal qubit on (x, y+1)
            self.qubit_index(x, y, 1)                # Vertical qubit on (x, y)
        ]
        xops = list(filter(lambda x: x != None, indices))
        xstring = sparse_to_paulistring(xops, self.num_qubits + nancillas, 'X')
        if tc == 0:
            return xstring
        else:
            return xops

    def hamiltonian(self, J=1,nancillas=0):
        allstars = [self.star_operator(i,j,nancillas) for i in range(self.Lx) for j in range(self.Ly)]
        allplaquettes = [self.plaquette_operator(i,j,nancillas) for i in range(self.Lx-1) for j in range(self.Ly-1)]
        H = - SparsePauliOp(allstars) - SparsePauliOp(allplaquettes)
        return J*H
    
    def hamiltonian_tc(self, J=1, nancillas=0):
        allstars = [self.star_operator(i,j,nancillas,tc=1) for i in range(self.Lx) for j in range(self.Ly)]
        allplaquettes = [self.plaquette_operator(i,j,nancillas,tc=1) for i in range(self.Lx-1) for j in range(self.Ly-1)]
        a = array_to_tc_structure(allstars,self.num_qubits + nancillas,'Z')
        a.extend(array_to_tc_structure(allplaquettes,self.num_qubits + nancillas,'X'))
        weights = -1*np.ones(len(a))
        return a, weights

    def hamiltonian_tc_perturbation(self, h, nancillas=0):
        strs = array_to_tc_structure([[i] for i in range(self.num_qubits)], self.num_qubits + nancillas, 'Z')
        wts = -h*np.ones(len(strs),dtype=np.float64)
        return strs, wts

    def topo_ee_region(self):
        A = [self.qubit_index(i, j, 0) for i in range(self.Lx-1) for j in range(1,self.Ly//2)]
        A.extend([self.qubit_index(i, j, 1) for i in range(1,self.Lx-1) for j in range(self.Ly//2)])
        B = [self.qubit_index(i, j, 0) for i in range(self.Lx//2) for j in range(self.Ly//2,self.Ly-1)]
        B.extend([self.qubit_index(i, j, 1) for i in range(1,self.Lx//2) for j in range(self.Ly//2,self.Ly-1)])
        C = [self.qubit_index(i, j, 0) for i in range(self.Lx//2,self.Lx-1) for j in range(self.Ly//2,self.Ly-1)]
        C.extend([self.qubit_index(i, j, 1) for i in range(self.Lx//2, self.Lx-1) for j in range(self.Ly//2,self.Ly-1)])
        return A, B, C
    
    def groundstate(self, hamiltonian=None):
        if hamiltonian == None:
            H = self.hamiltonian().to_matrix(sparse=True)
        else:
            H = hamiltonian.to_matrix(sparse=True)
    
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H,1, which='SA',return_eigenvectors=True)
        return eigvals[0], eigvecs[:,0]

    def topo_ee(self, region=None, state=None):
        if state is None:
            gstate = state_to_qiskit(self.groundstate()[1])
        else:
            gstate = state_to_qiskit(state)
        
        if region is None:
            A, B, C = self.topo_ee_region()
        else:
            A, B, C = region
        all = set(range(self.num_qubits))
    
        Ac = all - set(A)
        Bc = all - set(B)
        Cc = all - set(C)

        ABc = all - set(A) - set(B)
        BCc = all - set(B) - set(C)
        CAc = all - set(A) - set(C)
        ABCc = all - set(A) - set(B) - set(C)

        rhoA = partial_trace(gstate, list(Ac))
        rhoB = partial_trace(gstate, list(Bc))
        rhoC = partial_trace(gstate, list(Cc))

        rhoAB = partial_trace(gstate, list(ABc))
        rhoBC = partial_trace(gstate, list(BCc))
        rhoCA = partial_trace(gstate, list(CAc))
        rhoABC = partial_trace(gstate, list(ABCc))

        return (entropy(rhoA) + entropy(rhoB) + entropy(rhoC) - entropy(rhoAB) - entropy(rhoBC) - entropy(rhoCA) +
                entropy(rhoABC))

#TODO: Move to notebook
"""
# Example usage
Lx, Ly = 4,3
toric_code = ToricCode(Lx, Ly)

for j in range(Ly):
    for i in range(Lx):
        print(f"Qubit index for ({i}, {j}, 0):", toric_code.qubit_index(i, j, 0))
        print(f"Qubit index for ({i}, {j}, 1):", toric_code.qubit_index(i, j, 1))

# Get the star operator indices at vertex (1, 2)
print("Star operator indices at (1, -):", toric_code.star_operator(1, 1))

# Get the plaquette operator indices at plaquette (1, 2)
# print("Plaquette operator indices at (1, 1):", toric_code.plaquette_operator(1, 1))
regions = toric_code.topo_ee_region()
H = toric_code.hamiltonian()
# sv = Statevector.from_label('0'*toric_code.num_qubits).evolve(refstate)
print("Hamiltonian: ", H)
# print("Energies ", scipy.sparse.linalg.eigs(H.to_matrix(sparse=True)))
state = toric_code.groundstate()
print("GS ", state)
print("Topological EE: ", toric_code.topo_ee())
"""

