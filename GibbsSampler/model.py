import pandas as pd


class Model:


    def __init__(self):

        self.data = None
        self.y_name = None
        self.initial_values = None
        self.priors = None
        self.variable_names = None


    def set_data(self, data, y_name):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Parameter 'data' must be an instance of 'pandas.DataFrame'")

        if not isinstance(y_name, str):
            raise TypeError("Parameter 'y_name' must be a string")

        if data.empty:
            raise ValueError("Parameter 'data' cannot be an empty 'pandas.DataFrame'")

        if y_name not in data.columns:
            raise ValueError(f"Column '{y_name}' not found in 'data'")

        self.data = data
        self.y_name = y_name


    def set_initial_values(self, values):

        if not isinstance(values, dict):
            raise TypeError("Parameter 'values' must be a dictionary")

        if len(values) == 0:
            raise ValueError("Parameter 'values' cannot be an empty dictionary")

        if 'intercept' not in values.keys():
            raise KeyError("Parameter 'values' must contain a 'intercept' key")

        self.initial_values = values


    def set_priors(self, priors):

        if not isinstance(priors, dict):
            raise TypeError("Parameter 'priors' must be a dictionary")

        if len(priors) == 0:
            raise ValueError("Parameter 'priors' cannot be an empty dictionary")

        if 'intercept' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'intercept' key")

        if 'sigma2' not in priors.keys():
            raise KeyError("Parameter 'priors' must contain a 'sigma2' key")

        for prior, values in priors.items():
            if not isinstance(values, dict):
                raise TypeError(f"The value of prior '{prior}' must be a dictionary")
            if len(values) == 0:
                raise ValueError(f"The value of prior '{prior}' cannot be an empty dictionary")
            if prior != 'sigma2':
                if set(values.keys()) != {'mean', 'variance'}:
                    raise ValueError(f"The value of prior '{prior}' must be a dictionary "
                                     f"containing 'mean' and 'variance' keys only")
            else:
                if set(values.keys()) != {'shape', 'scale'}:
                    raise ValueError(f"The value of prior '{prior}' must be a dictionary "
                                     f"containing 'shape' and 'scale' keys only")

        self.priors = priors
        self.variable_names = list(self.priors.keys())
        self.variable_names.insert(0, self.variable_names.pop(self.variable_names.index('intercept')))
