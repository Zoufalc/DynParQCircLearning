# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dynamic Parameterized Quantum Circuit Ansatz Generation."""

import tensorcircuit as tc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")

from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister, ClassicalRegister
from src.utilities.generate_toric_code_hamiltonian import *

def precompute(c):
    s = c.state()
    return tc.DMCircuit(c._nqubits, dminputs=s)

def cartanblock(params=None, paramindex = 0):
    qubit0, qubit1 = 0, 1
    blk = tc.Circuit(2)
    blk, paramindex = universalsingle(blk, qubit0, params, paramindex)
    blk, paramindex = universalsingle(blk, qubit1, params, paramindex)
    blk.rxx(qubit0, qubit1, theta=params[paramindex])
    blk.ryy(qubit0, qubit1, theta=params[paramindex+1])
    blk.rzz(qubit0, qubit1, theta=params[paramindex+2])
    paramindex = paramindex+3
    return blk, paramindex

def universalsingle(circuit, index, params, paramindex):
    circuit.ry(index, theta=params[paramindex])
    circuit.rz(index, theta=params[paramindex+1])
    circuit.ry(index, theta=params[paramindex + 2])
    return circuit, paramindex + 3

def onesetofunitaries(qc, claws, params, paramindex, measindx=None,measure=False,seed=None):
    for cl in claws:
        inst, paramindex = cartanblock(params, paramindex)
        qc.append(inst, indices=cl)
        if measure:
            if measindx is None:
                raise ValueError("Please supply an argument for `measindx`")
            locs = cl[0]
            kraus_ops = list(genreset_kraus(params[paramindex],params[paramindex+1],params[paramindex+2]))
            paramindex += 3
            qc = precompute(qc)
            qc.general_kraus(kraus_ops,[(locs,)]) # If using DMCircuit
            measindx += 1
            qc = precompute(qc)
            return qc, paramindex, measindx
        else:
            return qc, paramindex

def onelayerofsingleunitaries(qc, params, paramindex, n=None):
    if n is None:
        n = qc._nqubits

    for i in range(n):
        qc, paramindex = universalsingle(qc, i, params, paramindex)
    return qc, paramindex

def measurements(qc,plaquettes,params, paramindex, measindex):
    for p in plaquettes:
        qc.measure(p,measindex)
        with qc.if_test((measindex, 1)):
            qc.x(p)
        measindex += 1
    return qc, paramindex, measindex

def genreset_kraus(p, theta, phi):
    knothing = tc.gates.Gate(K.cos(p)*np.array([[1, 0], [0, 1]]))
    state = K.convert_to_tensor([[K.cos(theta)], [K.sin(theta)*K.exp(-1j*phi)]])
    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])
    k0 = tc.gates.Gate(K.sin(p)*(state @ zero.conj().transpose()))
    k1 = tc.gates.Gate(K.sin(p)*(state @ one.conj().transpose()))
    return knothing, k0, k1

def construct_circuit(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('θ', n)
    for j in range(n):
        qc.ry(params[j], j)
    for k in range(n-1):
        qc.cx(k, k+1)
    return params, qc


def construct_dyn_circuit_toriccodelattice(params,Lx,Ly,nlayers = None,howoften=3):
    toriccode = ToricCode(Lx,Ly)
    nplaquettes = (Lx-1)*(Ly-1)
    nq = 2*Lx*Ly - Lx - Ly 
    if nlayers is None:
        # nlayers = max(Lx,Ly)
        nlayers = 2
    qc = tc.Circuit(nq+nplaquettes + nplaquettes * (nlayers//howoften), split=split_conf)
    
    nmeasurements = nplaquettes * (nlayers//howoften)
    nparams = nplaquettes * 4 * 9 *nlayers + 0*nmeasurements + 3*nq
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")
    
    paramindex = 0

    claws = toriccode.all_claws_measurements()
    claws = [claws[i::4][j] for i in range(4) for j in range((Lx-1)*(Ly-1))] # Rearranges them so as to parallelise
    plaquettes = [toriccode.qubit_index(x,y,2) for x in range(Lx-1) for y in range(Ly-1)]
    measindex = 0
    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc,claws,params,paramindex)
        if l % howoften == howoften-1:
            for p in plaquettes:
                qc.cx(p,nq + nplaquettes + measindex)
                qc.cx(nq+nplaquettes+measindex, p)
                measindex += 1

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,nq)
    return qc

def test_ansatz(param, n, nlayers, n_resets):
    n = 2*n
    zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)
    c = tc.Circuit(n + n_resets)
    paramc = tc.backend.cast(param, tc.dtypestr)  # We assume the input param with dtype float64
    counter = 0
    for i in range(int(n/2)):
        c.ry(i, theta=paramc[counter])
        counter += 1
        c.cnot(i, i+int(n/2))
    for j in range(nlayers-1):
        for i in range(n - 1):
            c.exp1(i, i + 1, unitary=zz, theta=paramc[counter])
            counter += 1
        for i in range(n):
            c.rx(i, theta=paramc[counter])
            counter += 1
    # Intermediate layer of resets
    counter_aux = 0
    for i in np.random.randint(0, int(n/2), n_resets).tolist():
        c.cnot(i, n+counter_aux)
        c.cnot(n+counter_aux, i)
        counter_aux+=1
    # Final layer of rotations
    for i in range(n - 1):
        c.exp1(i, i + 1, unitary=zz, theta=paramc[counter])
        counter += 1
    for i in range(n):
        c.rx(i, theta=paramc[counter])
        counter += 1

    return paramc, c

split_conf = {
    "max_singular_values": 2,  # how many singular values are kept
    "fixed_choice": 1, # 1 for normal one, 2 for swapped one
}

def construct_dissipative_ansatz_genresets(n, nlayers, param=None):
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = (nlayers+0.5)*n*15 + 3*n_resets
    if param is None:
        import tensorflow as tf
        num_params = num_params.astype(int)
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        paramc = tc.backend.cast(param, tc.dtypestr)

    
    c = tc.Circuit(n + 2*n_resets, split=split_conf)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = range(n-1)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets; strength is a parameter as well
        for r in resets:
            if np.mod(r, 2) == 0:
                c.ry(int(n + counter_aux + n_resets), theta=paramc[param_counter]) # Strength of reset
                param_counter += 1 
                c.cnot(int(n + counter_aux + n_resets), int(n + counter_aux))
                c.ccx(int(r), int(n + counter_aux + n_resets), int(n + counter_aux))
                U = tc.gates.r_gate(theta=paramc[param_counter], alpha=paramc[param_counter+1], phi=0)
                c.multicontrol(int(n + counter_aux + n_resets), int(n + counter_aux), r, unitary=U,
                               ctrl=[int(n + counter_aux + n_resets), int(n + counter_aux)])
                counter_aux += 1
                param_counter += 2

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.ry(int(n + counter_aux + n_resets), theta=paramc[param_counter]) # Strength of reset
                param_counter += 1 
                c.cnot(int(n + counter_aux + n_resets), int(n + counter_aux))
                c.ccx(int(r), int(n + counter_aux + n_resets), int(n + counter_aux))
                U = tc.gates.r_gate(theta=paramc[param_counter], alpha=paramc[param_counter+1], phi=0)
                c.multicontrol(int(n + counter_aux + n_resets), int(n + counter_aux), r, unitary=U,
                               ctrl=[int(n + counter_aux + n_resets), int(n + counter_aux)])
                counter_aux += 1
                param_counter += 2

    # Coherent Layer
    # Even
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3
    return c


# Density matrix implementation
def construct_dissipative_ansatz_dm(n, nlayers, param=None, dm_varqte=False, return_param_counter=False):
    def kraus_depolarizing(p):
        p = jnp.abs(jnp.cos(p))  # Ensure p\in[0,1[
        K0 = jnp.sqrt(1 - p) * jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p / 3) * jnp.array([[0, 1], [1, 0]])
        K2 = jnp.sqrt(p / 3) * jnp.array([[0, 1j], [-1j, 0]])
        K3 = jnp.sqrt(p / 3) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3
    def kraus0(p, q):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        q = jnp.abs(jnp.cos(q))  # Ensure q\in[0,1[
        K0 = jnp.sqrt(1-p-q)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p) * jnp.array([[0, 0], [0, 1]])
        K2 = jnp.sqrt(p) * jnp.array([[0, 0], [1, 0]])
        K3 = jnp.sqrt(q) * jnp.array([[1, 0], [0, 0]])
        K4 = jnp.sqrt(q) * jnp.array([[0, 1], [0, 0]])
        return K0, K1, K2, K3, K4
    def kraus1(p): #Reset
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        K0 = jnp.sqrt(p)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(1-p)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(1-p)*jnp.array([[0, 1], [0, 0]])
        return K0, K1, K2
    def kraus2(p):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        K0 = jnp.array([[1, 0], [0, jnp.sqrt(1-p)]])
        K1 = jnp.array([[0, 0], [0, jnp.sqrt(p)]])
        return K0, K1
    def kraus3(p, r): #TODO
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        r = jnp.abs(jnp.cos(r))  # Ensure r\in[0,1[
        K0 = jnp.sqrt(1-p-r)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p / 2)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(p / 2)*jnp.array([[0, 1], [0, 0]])
        # K3 = jnp.sqrt(r / 2)*jnp.array([[0, 1], [1, 0]])
        K3 = jnp.sqrt(r) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3
    def kraus4(p, q, r): #TODO
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        q = jnp.abs(jnp.cos(q)) # Ensure q\in[0,1[
        r = jnp.abs(jnp.cos(r))  # Ensure r\in[0,1[
        K0 = jnp.sqrt(1-p-q-r)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p/2)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(p/2)*jnp.array([[0, 1], [0, 0]])
        K3 = jnp.sqrt(r/2)*jnp.array([[0, 1], [1, 0]])
        K4 = jnp.sqrt(q / 2) * jnp.array([[0, 0], [0, 1]])
        K5 = jnp.sqrt(q / 2) * jnp.array([[0, 0], [1, 0]])
        K6 = jnp.sqrt(r / 2) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3, K4, K5, K6
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int(nlayers * (12 * (n - 2) + 20 * n)*50)
    if param is None:
        paramc = np.random.rand(num_params)
        paramc = tc.backend.cast(paramc, tc.dtypestr)
    else:
        paramc = tc.backend.cast(param, tc.dtypestr)
    dmc = tc.DMCircuit(n)
    param_counter = 0
    for l in range(nlayers):
        resets = range(n - 1)
        # Even
        for i in range(n):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            dmc.cnot(i, i + 1)
            dmc.ry(i, theta=paramc[param_counter])
            dmc.rz(i, theta=paramc[param_counter + 1])
            dmc.cnot(i + 1, i)
            dmc.ry(i, theta=paramc[param_counter + 2])
            dmc.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets; strength is a parameter as well
        for r in resets:
            if np.mod(r, 2) == 0:
                K0, K1, K2 = kraus1(paramc[param_counter])  # Strength of reset
                dmc.general_kraus([K0, K1, K2], r)
                param_counter += 1
        # Odd
        for i in range(1, n - 1):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n - 1, 2):
            dmc.cnot(i, i + 1)
            dmc.ry(i, theta=paramc[param_counter])
            dmc.rz(i, theta=paramc[param_counter + 1])
            dmc.cnot(i + 1, i)
            dmc.ry(i, theta=paramc[param_counter + 2])
            dmc.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n - 1):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                K0, K1, K2 = kraus1(paramc[param_counter])  # Strength of reset
                dmc.general_kraus([K0, K1, K2], r)
                param_counter += 1

    # Coherent Layer
    # Even
    for i in range(n):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        dmc.cnot(i, i + 1)
        dmc.ry(i, theta=paramc[param_counter])
        dmc.rz(i, theta=paramc[param_counter + 1])
        dmc.cnot(i + 1, i)
        dmc.ry(i, theta=paramc[param_counter + 2])
        dmc.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        dmc.cnot(i, i + 1)
        dmc.ry(i, theta=paramc[param_counter])
        dmc.rz(i, theta=paramc[param_counter + 1])
        dmc.cnot(i + 1, i)
        dmc.ry(i, theta=paramc[param_counter + 2])
        dmc.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        dmc.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3

    if dm_varqte:

        for i in range(0, n):
            K0, K1, K2, K3 = kraus_depolarizing(paramc[param_counter])  # Strength of depolarization
            dmc.general_kraus([K0, K1, K2, K3], i)
            param_counter += 1

    if return_param_counter:
        return param_counter
    else:
        return dmc

def construct_dissipative_ansatz_genresetsDM(n, nlayers, param=None,seed=None):
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+1)*(n-1)*9) + 3*n_resets + 3*n
    if param is None:
        import tensorflow as tf
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if param.shape != (num_params,):
            raise ValueError('The length of the parameter array should be ', num_params, "You have ", param.shape)
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.densitymatrix.DMCircuit(n, split=split_conf)
    param_counter = 0
    measindx = 0
    claws = []
    for i in range(n-1):
            claws.append([i,i+1])
    claws = [claws[i::2][j] for i in range(2) for j in range(n//2 - i)]

    for l in range(nlayers):
        c, param_counter, measindx = onesetofunitaries(c,claws,param,param_counter,measindx,measure=True,seed=seed)
    for l in range(1):
        c, param_counter, measindx = onesetofunitaries(c,claws,param,param_counter,measindx,measure=False,seed=seed)
    
    c, param_counter = onelayerofsingleunitaries(c, param, param_counter,n)
    return c

def construct_dissipative_ansatz(n, nlayers, resets_nlayer, param=None, return_param_counter=False):
    n_resets = nlayers*resets_nlayer
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+0.5)*n*15)
    if param is None:
        import tensorflow as tf
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if len(param) != num_params:
            raise ValueError('The length of the parameter array should be ', num_params)
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.Circuit(n + n_resets)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = np.random.randint(0, n, resets_nlayer)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets
        for r in resets:
            if np.mod(r, 2) == 0:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

    #Coherent Layer
    # Even
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3


    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3
    if return_param_counter:
        return param_counter
    else:
        return c

def construct_simplified_dissipative_ansatz(n, nlayers, resets_nlayer, param=None):
    n_resets = nlayers*resets_nlayer
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+0.5)*n*15)
    if param is None:
        import tensorflow as tf
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if len(param) != num_params:
            raise ValueError('The length of the parameter array should be ', num_params, "You have ", len(param))
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.Circuit(n + n_resets)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = np.random.randint(0, n, resets_nlayer)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.cnot(i+1, i)
            param_counter += 1

        # Even Resets
        for r in resets:
            if np.mod(r, 2) == 0:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.cnot(i+1, i)
            param_counter += 1

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

    # Coherent even layer
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.cnot(i + 1, i)
        param_counter += 1
    return c

def construct_fldc_toriccodelattice(Lx,Ly):
    tc = ToricCode(Lx,Ly)
    nq = 2*Lx*Ly - Lx - Ly
    qr = QuantumRegister(nq)

    nblocks = (Lx-1)*(Ly-1)
    nparams = nblocks * 3 * 15

    params = ParameterVector('θ', nparams)
    paramindex = 0
    qc = QuantumCircuit(qr)
    claws = tc.all_claws()

    for cl in claws:
        inst, paramindex = cartanblock(params, paramindex)
        qc.append(inst, qargs=cl)

    return params, qc