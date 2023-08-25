import numpy as np


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flattens a matrix of dimensions ``(M, N)`` to ``(M*N, )``.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix with ``M`` rows and ``N`` columns.

    Returns
    -------
    numpy.ndarray
        Flatten array with ``M*N`` elements.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.array([[1, 2], [3, 4], [5, 6]])
    >>> a
    >>> array([[1, 2],
    ...        [3, 4],
    ...        [5, 6]])
    >>> flatten_matrix(a)
    >>> array([1, 2, 3, 4, 5, 6])
    """
    return np.asarray(matrix).reshape(-1)
