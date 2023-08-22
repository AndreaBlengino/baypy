from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):


    @abstractmethod
    def __init__(self):

        self.data = None
        self.response_variable = None
        self.initial_values = None
        self.priors = None
        self.variable_names = None


    @abstractmethod
    def set_data(self, data, response_variable):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

        if not isinstance(response_variable, str):
            raise TypeError("Parameter 'response_variable' must be a string")

        if data.empty:
            raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

        if response_variable not in data.columns:
            raise ValueError(f"Column '{response_variable}' not found in 'data'")


    @abstractmethod
    def set_initial_values(self, values):
        if not isinstance(values, dict):
            raise TypeError("Parameter 'values' must be a dictionary")

        if len(values) == 0:
            raise ValueError("Parameter 'values' cannot be an empty dictionary")

        if 'intercept' not in values.keys():
            raise KeyError("Parameter 'values' must contain a 'intercept' key")


    @abstractmethod
    def set_priors(self, priors):
        if not isinstance(priors, dict):
            raise TypeError("Parameter 'priors' must be a dictionary")

        if len(priors) == 0:
            raise ValueError("Parameter 'priors' cannot be an empty dictionary")

        if 'intercept' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'intercept' key")

        if 'variance' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'variance' key")

        for prior, values in priors.items():
            if not isinstance(values, dict):
                raise TypeError(f"The value of prior '{prior}' must be a dictionary")
            if len(values) == 0:
                raise ValueError(f"The value of prior '{prior}' cannot be an empty dictionary")
            if prior != 'variance':
                if set(values.keys()) != {'mean', 'variance'}:
                    raise KeyError(f"The value of prior '{prior}' must be a dictionary "
                                   f"containing 'mean' and 'variance' keys only")
            else:
                if set(values.keys()) != {'shape', 'scale'}:
                    raise KeyError(f"The value of prior '{prior}' must be a dictionary "
                                   f"containing 'shape' and 'scale' keys only")


class LinearModel(Model):


    def __init__(self):

        super().__init__()


    def set_data(self, data, response_variable):

        super().set_data(data = data, response_variable = response_variable)
        self.data = data
        self.response_variable = response_variable


    def set_initial_values(self, values):

        super().set_initial_values(values = values)
        self.initial_values = values


    def set_priors(self, priors):

        super().set_priors(priors = priors)
        self.priors = priors
        self.variable_names = list(self.priors.keys())
        self.variable_names.insert(0, self.variable_names.pop(self.variable_names.index('intercept')))
