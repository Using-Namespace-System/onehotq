import numpy as np

def statevector_from_composit_systems(amplitudes: np.ndarray) -> np.ndarray:
    if(amplitudes.size>=4):
        return np.tensordot(amplitudes[[0,1]],statevector_from_composit_systems(amplitudes[2:]),axes=0).flatten()
    else:
        return amplitudes
    
def compose(amplitudes, subsys_dims):
    pass

def expand_small_operator(op,system,subsys):
    identity = np.diag(np.ones(system.shape[0]))
    return np.kron(op,identity)

def operator(op, system, subsys_dims):
    return op @ system