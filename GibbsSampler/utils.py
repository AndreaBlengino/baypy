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


def matrix_to_frame(posteriors: dict) -> pd.DataFrame:
    frame = pd.DataFrame()
    for posterior, posterior_samples in posteriors.items():
        frame[posterior] = flatten_matrix(posterior_samples)

    return frame
