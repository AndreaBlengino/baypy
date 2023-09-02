import baypy as bp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv(r'data/data.csv')
data.drop(columns = ['No'], inplace = True)
data.columns = [' '.join(col.split(' ')[1:]) for col in data.columns]
data.rename(columns = {'distance to the nearest MRT station': 'MRT station distance',
                       'number of convenience stores': 'stores number',
                       'house price of unit area': 'house price'},
            inplace = True)


pd.plotting.scatter_matrix(frame = data, figsize = (10, 10))

plt.tight_layout()

plt.show()


data = data[(data['house price'] > 8) & (data['house price'] < 115)]
data['log house price'] = np.log(data['house price'])
data['log MRT station distance'] = np.log(data['MRT station distance'])


model = bp.model.LinearModel()
model.data = data
model.response_variable = 'log house price'
model.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                'transaction date': {'mean': 0, 'variance': 1e6},
                'house age': {'mean': 0, 'variance': 1e6},
                'log MRT station distance': {'mean': 0, 'variance': 1e6},
                'stores number': {'mean': 0, 'variance': 1e6},
                'latitude': {'mean': 0, 'variance': 1e6},
                'longitude': {'mean': 0, 'variance': 1e6},
                'variance': {'shape': 1, 'scale': 1e-6}}

sampler = bp.regression.LinearRegression(model = model)
posteriors = sampler.sample(n_iterations = 1000, burn_in_iterations = 50, n_chains = 3, seed = 137)

bp.diagnostics.effective_sample_size(posteriors = posteriors)
bp.diagnostics.autocorrelation_summary(posteriors = posteriors)
bp.diagnostics.autocorrelation_plot(posteriors = posteriors)

bp.analysis.trace_plot(posteriors = posteriors)
bp.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'log house price')
bp.analysis.summary(posteriors = posteriors)
