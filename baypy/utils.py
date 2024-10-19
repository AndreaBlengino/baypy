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

    .. admonition:: Raises
       :class: warning

       TypeError
           If ``matrix`` is not a ``numpy.ndarray``.
       ValueError
           If ``matrix`` is an empty ``numpy.ndarray``.

    .. admonition:: Examples
       :class: important

       >>> import numpy
       >>> a = numpy.array([[1, 2], [3, 4], [5, 6]])
       >>> a
       array([[1, 2],
              [3, 4],
              [5, 6]])
       >>> flatten_matrix(a)
       array([1, 2, 3, 4, 5, 6])
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(
            "Parameter 'matrix' must be an instance of 'numpy.ndarray'"
        )

    if matrix.size == 0:
        raise ValueError("Parameter 'matrix' is empty")

    return np.asarray(matrix).reshape(-1)


def matrices_to_frame(matrices_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    """Organizes a dictionary of matrices in a ``pandas.DataFrame``. Each
    matrix becomes a frame column, with column name equal to the matrix'
    relative key in the dictionary. If the matrix has dimensions ``(M, N``),
    then the relative frame column has length ``M*N``.

    Parameters
    ----------
    matrices_dict : dict
        Dictionary of matrices to be organized. Matrices and relative names are
        key-value pairs. Each matrix is a ``numpy.ndarray`` with dimensions
        ``(M, N)``.

    Returns
    -------
    pandas.DataFrame
        Reorganized matrices frame. Matrices are organized in a
        ``pandas.DataFrame``, one for each column. The length of the frame is
        ``M*N``.

    .. admonition:: Raises
       :class: warning

       TypeError
           - If ``matrices_dict`` is not a ``dict``,
           - if a ``matrices_dict`` value is not a ``numpy.ndarray``.
       ValueError
           If a ``matrices_dict`` value is an empty ``numpy.ndarray``.

    .. admonition:: Examples
       :class: important

       >>> import numpy
       >>> a = numpy.array([[1, 2], [3, 4], [5, 6]])
       >>> b = numpy.array([[7, 8], [9, 10], [11, 12]])
       >>> d = {'a': a, 'b': b}
       >>> matrices_to_frame(d)
          a   b
       0  1   7
       1  2   8
       2  3   9
       3  4  10
       4  5  11
       5  6  12
    """
    if not isinstance(matrices_dict, dict):
        raise TypeError("Parameter 'matrices_dict' must be a dictionary")

    for matrix_name, matrix in matrices_dict.items():
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f"Matrix '{matrix_name}' must be an instance of "
                f"'numpy.ndarray'"
            )

        if matrix.size == 0:
            raise ValueError(f"Matrix '{matrix_name}' is empty")

    return pd.DataFrame(
        {col: flatten_matrix(matrix) for col, matrix in matrices_dict.items()}
    )


def dot_product(
    data: pd.DataFrame,
    regressors: dict[str, float | int]
) -> np.ndarray:
    """Computes the dot product between columns of ``data`` and values of
    ``regressors``.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataframe to be used for the dot product. It cannot be empty. It
        must contain all ``regressors`` keys.

    regressors : dict
        Dictionary with regressors' values. It cannot be empty. All regressors
        and relative values must be a key-value pair.

    Returns
    -------
    numpy.ndarray
        Array of computed dot product. Each element is the dot product of a
        ``data`` row with respect to each ``regressors``. It has the same
        length of ``data``.

    .. admonition:: Raises
       :class: warning

       TypeError
           - If ``data`` is not a ``pandas.DataFrame``,
           - if ``regressors`` is not a ``dict``,
           - if a ``regressors`` value is not a ``int`` or a ``float``.
       KeyError
           If a ``regressors`` key is not a column of ``data``.
       ValueError
           - If ``data`` is an empty ``pandas.DataFrame``,
           - if ``regressors`` is an empty ``dict``.

    .. admonition:: Examples
       :class: important

       >>> import pandas as pd
       >>> data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
       >>> regressors = {'a': 2, 'b': -1}
       >>> dot_product(data = data, regressors = regressors)
       array([-2, -1,  0])
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Parameter 'data' must be an instance of 'pandas.DataFrame'"
        )

    if data.empty:
        raise ValueError(
            "Parameter 'data' cannot be an empty 'pandas.DataFrame'"
        )

    if not isinstance(regressors, dict):
        raise TypeError("Parameter 'regressors' must be a dictionary")

    if len(regressors) == 0:
        raise ValueError(
            "Parameter 'regressors' cannot be an empty dictionary"
        )

    for regressor in regressors.keys():
        if regressor not in data.columns:
            raise KeyError(f"Column '{regressor}' not found in 'data'")

    if not all(
        [
            isinstance(regressor, float | int)
            for regressor in regressors.values()
        ]
    ):
        raise TypeError("All 'regressors' values must be integers or float.")

    data = data[regressors.keys()]
    return np.dot(data, list(regressors.values()))
