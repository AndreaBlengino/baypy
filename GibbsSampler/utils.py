import numpy as np
import pandas as pd


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


def matrices_to_frame(matrices_dict: dict) -> pd.DataFrame:
    """Organizes a dictionary of matrices in a ``pandas.DataFrame``. Each matrix becomes a frame column, with column
    name equal to the matrix' relative key in the dictionary. It the matrix has dimensions ``(M, N``), then the relative
    frame column has length ``M*N``.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary of matrices to be organized. Matrices and relative names are key-value pairs. Each matrix is a
        ``numpy.ndarray`` with dimensions ``(M, N)``.

    Returns
    -------
    pandas.DataFrame
        Reorganized matrices frame. Matrices are organized in a ``pandas.DataFrame``, one for each columns. The
        length of the frame is ``M*N``.

    Examples
    --------
    >>> import numpy
    >>> a = numpy.array([[1, 2], [3, 4], [5, 6]])
    >>> b = numpy.array([[7, 8], [9, 10], [11, 12]])
    >>> d = {'a': a, 'b': b}
    >>> matrices_to_frame(d)
    >>>    a   b
    >>> 0  1   7
    >>> 1  2   8
    >>> 2  3   9
    >>> 3  4  10
    >>> 4  5  11
    >>> 5  6  12
    """
    return pd.DataFrame({col: flatten_matrix(matrix) for col, matrix in matrices_dict.items()})
