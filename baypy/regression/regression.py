from abc import ABC, abstractmethod
from baypy.model import Model
import pandas as pd


class Regression(ABC):


    @abstractmethod
    def __init__(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError(f"Parameter 'model' must be an instance of '{Model.__module__}.{Model.__name__}'")

        if model.data is None:
            raise ValueError("Missing 'data' in 'model'")

        if model.response_variable is None:
            raise ValueError("Missing 'response_variable' in 'model'")

        if model.response_variable not in model.data.columns:
            raise ValueError(f"Column '{model.response_variable}' not found in 'data'")

        if model.priors is None:
            raise ValueError("Missing 'priors' in 'model'")

        for prior in model.priors.keys():
            if (prior not in  ['intercept', 'variance']) and (prior not in model.data.columns):
                raise ValueError(f"Column '{prior}' not found in 'Model.data'")

        self.model = model
        self.posteriors = None


    @abstractmethod
    def sample(self, n_iterations: int, burn_in_iterations: int, n_chains: int, seed: int = None) -> dict:
        if not isinstance(n_iterations, int):
            raise TypeError("Parameter 'n_iteration' must be an integer")

        if not isinstance(burn_in_iterations, int):
            raise TypeError("Parameter 'burn_in_iterations' must be an integer")

        if not isinstance(n_chains, int):
            raise TypeError("Parameter 'n_chains' must be an integer")

        if (not isinstance(seed, int)) and (seed is not None):
            raise TypeError("Parameter 'seed' must be an integer")

        if n_iterations <= 0:
            raise ValueError("Parameter 'n_iteration' must be greater than 0")

        if burn_in_iterations < 0:
            raise ValueError("Parameter 'burn_in_iterations' must be greater than or equal to 0")

        if n_chains <= 0:
            raise ValueError("Parameter 'n_chains' must be greater than 0")

        if seed is not None:
            if (seed < 0) or (seed > 2**32 - 1):
                raise ValueError("Parameter 'seed' must be between 0 and 2**32 - 1")


    @abstractmethod
    def posteriors_to_frame(self) -> pd.DataFrame:
        if self.posteriors is None:
            raise ValueError("Posteriors not available, run 'LinearRegression.sample' to generate posteriors")
