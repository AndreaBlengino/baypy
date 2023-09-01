import baypy as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'data/data.csv')


fig_1, ax_1 = plt.subplots()

ax_1.plot(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '', alpha = 0.5)

ax_1.set_xlabel('TV')
ax_1.set_ylabel('Sales')

ax_1.tick_params(bottom = False, top = False, left = False, right = False)

plt.show()


model_1 = bp.model.LinearModel()

model_1.data = data
model_1.response_variable = 'Sales'
model_1.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                  'TV': {'mean': 0, 'variance': 1e6},
                  'variance': {'shape': 1, 'scale': 1e-6}}

regression_1 = bp.regression.LinearRegression(model = model_1)
posteriors_1 = regression_1.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3, seed = 137)


bp.diagnostics.effective_sample_size(posteriors = posteriors_1)
bp.diagnostics.autocorrelaion_summary(posteriors = posteriors_1)
bp.diagnostics.autocorrelation_plot(posteriors = posteriors_1)

bp.analysis.trace_plot(posteriors = posteriors_1)
bp.analysis.residuals_plot(posteriors = posteriors_1, data = data, response_variable = 'Sales')


data_tmp = pd.DataFrame()
for posterior, posterior_sample in posteriors_1.items():
    data_tmp[posterior] = np.asarray(posterior_sample).reshape(-1)
data_tmp['error 1'] = np.random.normal(loc = 0, scale = np.sqrt(data_tmp['variance']), size = len(data_tmp))

x = np.linspace(data['TV'].min(), data['TV'].max(), 50)


fig_2, ax_2 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_2.plot(x, y, color = 'blue', linewidth = 1, alpha = 0.1)
ax_2.plot(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '',
          markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_2.set_xlabel('TV')
ax_2.set_ylabel('Sales')
ax_2.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


fig_3, ax_3 = plt.subplots()

ax_3.loglog(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '', alpha = 0.5)

ax_3.set_xlabel('TV')
ax_3.set_ylabel('Sales')

plt.show()


data['log TV'] = np.log(data['TV'])
data['log Sales'] = np.log(data['Sales'])

model_2 = bp.model.LinearModel()

model_2.data = data
model_2.response_variable = 'log Sales'
model_2.priors = {'intercept': {'mean': 0, 'variance': 1e6},
                  'log TV': {'mean': 0, 'variance': 1e6},
                  'variance': {'shape': 1, 'scale': 1e-6}}

regression_2 = bp.regression.LinearRegression(model = model_2)
posteriors_2 = regression_2.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3, seed = 137)


bp.diagnostics.effective_sample_size(posteriors = posteriors_2)
bp.diagnostics.autocorrelation_summary(posteriors = posteriors_2)
bp.diagnostics.autocorrelation_plot(posteriors = posteriors_2)

bp.analysis.trace_plot(posteriors = posteriors_2)
bp.analysis.residuals_plot(posteriors = posteriors_2, data = data, response_variable = 'log Sales')
bp.analysis.summary(posteriors = posteriors_2)


data_tmp = pd.DataFrame()
for posterior, posterior_sample in posteriors_2.items():
    data_tmp[posterior] = np.asarray(posterior_sample).reshape(-1)
data_tmp['error 2'] = np.random.normal(loc = 0, scale = np.sqrt(data_tmp['variance']), size = len(data_tmp))

x = np.linspace(data['log TV'].min(), data['log TV'].max(), 50)


fig_4, ax_4 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_4.plot(x, y, color = 'blue', linewidth = 1, alpha = 0.1)
ax_4.plot(data['log TV'].values, data['log Sales'].values, marker = 'o', linestyle = '',
        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_4.set_xlabel('log TV')
ax_4.set_ylabel('log Sales')
ax_4.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


fig_5, ax_5 = plt.subplots()

for row in zip(*data_tmp.to_dict('list').values()):
    y = row[0] + row[1]*x + row[3]
    ax_5.plot(np.exp(x), np.exp(y), color = 'blue', linewidth = 1, alpha = 0.1)
ax_5.plot(data['TV'].values, data['Sales'].values, marker = 'o', linestyle = '',
        markerfacecolor = 'none', markeredgecolor = 'red', markeredgewidth = 1.2)

ax_5.set_xlabel('TV')
ax_5.set_ylabel('Sales')
ax_5.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()


bp.analysis.compute_DIC(posteriors = posteriors_1, data = data, response_variable = 'Sales')
bp.analysis.compute_DIC(posteriors = posteriors_2, data = data, response_variable = 'log Sales')
