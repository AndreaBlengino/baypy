from baypy.model import LinearModel
from baypy.regression import LinearRegression
import baypy as bp
import pandas as pd


data = pd.read_csv(r'data/data.csv')


model = LinearModel()

model.data = data
model.response_variable = 'heart disease'
model.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                'biking': {'mean': 0, 'variance': 1e9},
                'smoking': {'mean': 0, 'variance': 1e9},
                'variance': {'shape': 1, 'scale': 1e-9}}

LinearRegression.sample(model = model, n_iterations = 500, burn_in_iterations = 50, n_chains = 3, seed = 137)


bp.diagnostics.effective_sample_size(posteriors = model.posteriors)
bp.diagnostics.autocorrelation_summary(posteriors = model.posteriors)
bp.diagnostics.autocorrelation_plot(posteriors = model.posteriors)

bp.analysis.trace_plot(posteriors = model.posteriors)
bp.analysis.residuals_plot(model = model)
bp.analysis.summary(posteriors = model.posteriors)
