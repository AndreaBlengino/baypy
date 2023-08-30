import baypy as bp
import pandas as pd


data = pd.read_csv(r'data/data.csv')


model = bp.model.LinearModel()

model.data = data
model.response_variable = 'heart disease'
model.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                'biking': {'mean': 0, 'variance': 1e9},
                'smoking': {'mean': 0, 'variance': 1e9},
                'variance': {'shape': 1, 'scale': 1e-9}}

regression = bp.regression.LinearRegression(model = model)
posteriors = regression.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3, seed = 137)


bp.diagnostics.effective_sample_size(posteriors = posteriors)
bp.diagnostics.autocorrelation_summary(posteriors = posteriors)
bp.diagnostics.autocorrelation_plot(posteriors = posteriors)

bp.analysis.trace_plot(posteriors = posteriors)
bp.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'heart disease')
bp.analysis.summary(posteriors = posteriors)
bp.analysis.compute_DIC(posteriors = posteriors, data = data, response_variable = 'heart disease')
