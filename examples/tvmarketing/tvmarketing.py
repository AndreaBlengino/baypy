import GibbsSampler as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(137)
data = pd.read_csv(r'data.csv')


fig_1, ax_1 = plt.subplots()

ax_1.plot(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '', alpha = 0.5)

ax_1.set_xlabel('TV')
ax_1.set_ylabel('Sales')

ax_1.tick_params(bottom = False, top = False, left = False, right = False)

plt.show()


fig_2, ax_2 = plt.subplots()

ax_2.loglog(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '', alpha = 0.5)

ax_2.set_xlabel('TV')
ax_2.set_ylabel('Sales')

plt.show()


data['log TV'] = np.log(data['TV'])
data['log Sales'] = np.log(data['Sales'])


model = gs.model.LinearModel()

model.data = data
model.response_variable = 'log Sales'
model.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                'log TV': {'mean': 0, 'variance': 1e6},
                'variance': {'shape': 1, 'scale': 1e-6}}

regression = gs.regression.LinearRegression(model = model)
posteriors = regression.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3)


gs.diagnostics.effective_sample_size(posteriors = posteriors)
gs.diagnostics.autocorrelation_summary(posteriors = posteriors)
gs.diagnostics.autocorrelation_plot(posteriors = posteriors)

gs.analysis.trace_plot(posteriors = posteriors)
gs.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'log Sales')
gs.analysis.summary(posteriors = posteriors)
gs.analysis.compute_DIC(posteriors = posteriors, data = data, response_variable = 'log Sales')


data_tmp = pd.DataFrame()
for posterior, posterior_sample in posteriors.items():
    data_tmp[posterior] = np.asarray(posterior_sample).reshape(-1)
data_tmp['error'] = np.random.normal(loc = 0, scale = np.sqrt(data_tmp['variance']), size = len(data_tmp))

x = np.linspace(data['log TV'].min(), data['log TV'].max(), 50)


fig_3, ax_3 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_3.plot(x, y, color = 'blue', linewidth = 1, alpha = 0.1)
ax_3.plot(data['log TV'].values, data['log Sales'].values, marker = 'o', linestyle = '',
        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_3.set_xlabel('log TV')
ax_3.set_ylabel('log Sales')
ax_3.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


fig_4, ax_4 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_4.plot(np.exp(x), np.exp(y), color = 'blue', linewidth = 1, alpha = 0.1)
ax_4.plot(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '',
        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_4.set_xlabel('TV')
ax_4.set_ylabel('Sales')
ax_4.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()
