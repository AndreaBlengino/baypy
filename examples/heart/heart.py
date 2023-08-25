import GibbsSampler as gs
import pandas as pd
import numpy as np

np.random.seed(137)
data = pd.read_csv(r'data.csv')


model = gs.model.LinearModel()

model.data = data
model.response_variable = 'heart disease'
model.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                'biking': {'mean': 0, 'variance': 1e9},
                'smoking': {'mean': 0, 'variance': 1e9},
                'variance': {'shape': 1, 'scale': 1e-9}}

regression = gs.regression.LinearRegression(model = model)
posteriors = regression.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3)


gs.diagnostics.effective_sample_size(posteriors = posteriors)
gs.diagnostics.autocorrelation_summary(posteriors = posteriors)
gs.diagnostics.autocorrelation_plot(posteriors = posteriors)

gs.analysis.trace_plot(posteriors = posteriors)
gs.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'heart disease')
gs.analysis.summary(posteriors = posteriors)
gs.analysis.compute_DIC(posteriors = posteriors, data = data, response_variable = 'heart disease')
