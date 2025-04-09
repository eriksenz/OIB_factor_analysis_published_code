import numpy as np

# define function to calculate orthonormal basis for an input compositional matrix
def construct_orthonormal_basis(x):
    """
    Constructs the base for the (d-1) simplex vector space using the Egozcue (2005) equations.
    
    Parameters:
    x (array): compositional matrix (observations x variables).
    """
    # extract number of variables from input matrix
    d = np.shape(x)[1]
    dim = d - 1
    
    V = np.zeros((d, dim))
    for vi in range(dim):
        for idx in range(vi + 1):
            V[idx, vi] = np.sqrt((vi + 1) / (vi + 2)) * (1 / (vi + 1))
        if vi < d - 1:
            V[vi + 1, vi] = np.sqrt((vi + 1) / (vi + 2)) * (-1)
    return V

# define function to execute ilr transform on an input compositional matrix
def ilr_transform(x):
    """
    Perform the isometric log-ratio (ilr) transformation on the input matrix.

    Parameters:
    x (array): compositional matrix (observations x variables).

    Returns:
    numpy.ndarray: A matrix with the ilr-transformed values, having n rows and (d-1) columns.
    """
    n_rows, n_cols = x.shape
    x_ilr = np.empty((n_rows, n_cols - 1))
    
    for i in range(1, n_cols):
        # Calculate the geometric mean of the first i columns for each row
        geometric_mean = np.prod(x[:, :i], axis=1) ** (1 / i)
        
        # Compute the ilr transformation
        x_ilr[:, i - 1] = np.sqrt(i / (i + 1)) * np.log(geometric_mean / x[:, i])
    
    return x_ilr
    