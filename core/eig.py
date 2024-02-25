#each point from largest to smallest has its own eigenvec with a one in its index from bottom to top
#remaining eigenvecs are a diagonal starting in lower right above the smallest point's egenvec skipping point indexes
#from bottom to top there is consecutive eigenvals of one for each point

def gpu_eigvals_diag_matrix(dense_points,num_qubits):
    import cupy as cp
    return cp.sparse.csr_matrix(
        (cp.ones(dense_points.size),
        (cp.arange(int(2**num_qubits)-1, (int(2**num_qubits)-1)-dense_points.size, -1),
        cp.arange(int(2**num_qubits)-1, (int(2**num_qubits)-1)-dense_points.size, -1))),
        shape=(int(2**num_qubits),int(2**num_qubits)),dtype=cp.float32)

def gpu_eigvals_diag(dense_points, num_qubits):
    import cupy as cp
    eigvals_diag = cp.zeros(int(2**num_qubits))
    eigvals_diag[-dense_points.size:] += 1
    return eigvals_diag

def gpu_eigendecomp(dense_point_indexes, num_qubits):
    import cupy as cp
    row = cp.arange(int(2**num_qubits))
    col = cp.array(
        dense_point_indexes + [x for x in np.arange(int(2**num_qubits)-1, -1, -1) if x not in dense_point_indexes],dtype=int)[::-1]
    data = cp.ones(int(2**num_qubits), dtype=int)
    return cp.sparse.csr_matrix(
        (data,(row,col)),
        shape=(int(2**num_qubits),int(2**num_qubits)),dtype=cp.float32)

def gpu_eig(dense_points, num_qubits):
    dense_point_indexes = (dense_points - 1).tolist()
    return (
        gpu_eigendecomp(dense_point_indexes,num_qubits),
        gpu_eigvals_diag(dense_points, num_qubits),
        gpu_eigvals_diag_matrix(dense_points,num_qubits)
    )

def cpu_eigendecomp(dense_point_indexes,num_qubits):
    import numpy as np
    import scipy as sp
    row = np.arange(int(2**num_qubits))
    col = np.array(
        dense_point_indexes + [x for x in np.arange(int(2**num_qubits)-1, -1, -1) if x not in dense_point_indexes],dtype=int)[::-1]
    data = np.ones(int(2**num_qubits), dtype=int)
    return sp.sparse.csr_array(
        (data,(row,col)),
        shape=(int(2**num_qubits),int(2**num_qubits)),dtype=np.float32)

def cpu_eigvals_diag_matrix(dense_points,num_qubits):
    import numpy as np
    import scipy as sp
    return sp.sparse.csr_matrix(
        (np.ones(dense_points.size),
        (np.arange(int(2**num_qubits)-1, (int(2**num_qubits)-1)-dense_points.size, -1),
        np.arange(int(2**num_qubits)-1, (int(2**num_qubits)-1)-dense_points.size, -1))),
        shape=(int(2**num_qubits),int(2**num_qubits)),dtype=np.float32)

def cpu_eigvals_diag(dense_points, num_qubits):
    import numpy as np
    eigvals_diag = np.zeros(int(2**num_qubits))
    eigvals_diag[-dense_points.size:] += 1
    return eigvals_diag

def cpu_eig(dense_points, num_qubits):
    dense_point_indexes = (dense_points - 1).tolist()
    return (
        cpu_eigendecomp(dense_point_indexes,num_qubits),
        cpu_eigvals_diag(dense_points, num_qubits),
        cpu_eigvals_diag_matrix(dense_points,num_qubits)
    )
