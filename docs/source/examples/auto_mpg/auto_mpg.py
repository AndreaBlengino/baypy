from baypy.model import LinearModel
from baypy.regression import LinearRegression
import baypy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'data/data.csv')
data.dropna(inplace=True)


pd.plotting.scatter_matrix(frame=data, figsize=(10, 10))

plt.tight_layout()

plt.show()


data['log mpg'] = np.log(data['mpg'])
data['log weight'] = np.log(data['weight'])


model = LinearModel()
model.data = data
model.response_variable = 'log mpg'
model.priors = {
    'intercept': {'mean': 0, 'variance': 1e6},
    'cylinders': {'mean': 0, 'variance': 1e6},
    'log weight': {'mean': 0, 'variance': 1e6},
    'acceleration': {'mean': 0, 'variance': 1e6},
    'model year': {'mean': 0, 'variance': 1e6},
    'variance': {'shape': 1, 'scale': 1e-6}
}

LinearRegression.sample(
    model=model,
    n_iterations=1000,
    burn_in_iterations=50,
    n_chains=3,
    seed=137
)

bp.diagnostics.effective_sample_size(posteriors=model.posteriors)
bp.diagnostics.autocorrelation_summary(posteriors=model.posteriors)
bp.diagnostics.autocorrelation_plot(posteriors=model.posteriors)

bp.analysis.trace_plot(posteriors=model.posteriors)
bp.analysis.residuals_plot(model=model)
bp.analysis.summary(posteriors=model.posteriors)
