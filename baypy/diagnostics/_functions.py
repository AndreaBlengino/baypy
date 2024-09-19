import numpy as np


def _compute_autocorrelation(vector: np.ndarray, max_lags: int) -> np.ndarray:

    normalized_vector = vector - vector.mean()
    autocorrelation = np.correlate(
        normalized_vector,
        normalized_vector, 'full'
    )[len(normalized_vector) - 1:]
    autocorrelation = autocorrelation/vector.var()/len(vector)
    autocorrelation = autocorrelation[:max_lags]

    return autocorrelation
