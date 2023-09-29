from baypy.model import LinearModel
from baypy.regression import LinearRegression
import baypy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'data/data.csv')


model = LinearModel()

model.data = data
model.response_variable = 'Salary'
model.priors = {'intercept': {'mean': 0, 'variance': 1e12},
                'YearsExperience': {'mean': 0, 'variance': 1e12},
                'variance': {'shape': 1, 'scale': 1e-12}}

LinearRegression.sample(model = model, n_iterations = 5000, burn_in_iterations = 50, n_chains = 3, seed = 137)


bp.diagnostics.effective_sample_size(posteriors = model.posteriors)
bp.diagnostics.autocorrelation_summary(posteriors = model.posteriors)
bp.diagnostics.autocorrelation_plot(posteriors = model.posteriors)

bp.analysis.trace_plot(posteriors = model.posteriors)
bp.analysis.residuals_plot(model = model)
bp.analysis.summary(posteriors = model.posteriors)


distribution = model.predict_distribution(predictors = {'YearsExperience': 5})

fig_2, ax_2 = plt.subplots()

ax_2.hist(distribution, bins = int(np.sqrt(len(distribution))), color = 'blue', alpha = 0.5, density = True)

ax_2.set_xlabel('Salary')
ax_2.set_ylabel('Probability Density')
ax_2.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


posteriors_data = pd.DataFrame()
for posterior, posterior_sample in model.posteriors.items():
    posteriors_data[posterior] = np.asarray(posterior_sample).reshape(-1)
posteriors_data['error'] = np.random.normal(loc = 0, scale = np.sqrt(posteriors_data['variance']), size = len(posteriors_data))

years_experience = np.linspace(data['YearsExperience'].min(), data['YearsExperience'].max(), 50)


fig_1, ax_1 = plt.subplots()

for row in zip(*posteriors_data.to_dict('list').values()):
    salary = row[0] + row[1]*years_experience + row[3]
    ax_1.plot(years_experience, salary, color = 'blue', linewidth = 1, alpha = 0.1)
ax_1.plot(data['YearsExperience'].values, data['Salary'].values, marker = 'o', linestyle = '',
          markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_1.set_xlabel('YearsExperience')
ax_1.set_ylabel('Salary')
ax_1.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()
