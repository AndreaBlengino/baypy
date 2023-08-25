import GibbsSampler as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


np.random.seed(137)
data = pd.read_csv(r'data.csv')


model = gs.model.LinearModel()

model.data = data
model.response_variable = 'Salary'
model.priors = {'intercept': {'mean': 0, 'variance': 1e12},
                'YearsExperience': {'mean': 0, 'variance': 1e12},
                'variance': {'shape': 1, 'scale': 1e-12}}

regression = gs.regression.LinearRegression(model = model)
posteriors = regression.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3)


gs.diagnostics.effective_sample_size(posteriors = posteriors)
gs.diagnostics.autocorrelation_summary(posteriors = posteriors)
gs.diagnostics.autocorrelation_plot(posteriors = posteriors)

gs.analysis.trace_plot(posteriors = posteriors)
gs.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'Salary')
gs.analysis.summary(posteriors = posteriors)
gs.analysis.compute_DIC(posteriors = posteriors, data = data, response_variable = 'Salary')


data_tmp = pd.DataFrame()
for posterior, posterior_sample in posteriors.items():
    data_tmp[posterior] = np.asarray(posterior_sample).reshape(-1)
data_tmp['error'] = np.random.normal(loc = 0, scale = np.sqrt(data_tmp['variance']), size = len(data_tmp))

x = np.linspace(data['YearsExperience'].min(), data['YearsExperience'].max(), 50)


fig_1, ax_1 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_1.plot(x, y, color = 'blue', linewidth = 1, alpha = 0.1)
ax_1.plot(data['YearsExperience'].values, data['Salary'].values, marker = 'o', linestyle = '',
        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_1.set_xlabel('YearsExperience')
ax_1.set_ylabel('Salary')
ax_1.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


distribution = gs.analysis.predict_distribution(posteriors = posteriors, predictors = {'YearsExperience': 5})

fig_2, ax_2 = plt.subplots()

ax_2.hist(distribution, bins = int(np.sqrt(len(distribution))), color = 'blue', alpha = 0.5, density = True)

ax_2.set_xlabel('Salary')
ax_2.set_ylabel('Probability Density')
ax_2.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()
