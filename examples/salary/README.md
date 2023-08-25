### Model Set Up

Pretending to fit the salary dataset:

```python
import numpy as np
import pandas as pd

np.random.seed(137)
data = pd.read_csv(r'data.csv')
```

Setting-up a linear regression model, using non-informative priors for
regressors and variance:

```python
import GibbsSampler as gs

model = gs.model.LinearModel()
model.data = data
model.response_variable = 'Salary'
model.priors = {'intercept': {'mean': 0, 'variance': 1e12},
                'YearsExperience': {'mean': 0, 'variance': 1e12},
                'variance': {'shape': 1, 'scale': 1e-12}}
```

### Sampling

Run the regression sampling on 3 Markov chains and discarding the first 
burn-in draws:

```python
regression = gs.regression.LinearRegression(model = model)
posteriors = regression.sample(n_iterations = 500, burn_in_iterations = 50, n_chains = 3)
```

### Convergence Diagnostics

Asses the model convergence diagnostics:

```python
gs.diagnostics.effective_sample_size(posteriors = posteriors)

                       intercept  YearsExperience  variance
Effective Sample Size    1336.25          1328.85   1360.84
```

```python
gs.diagnostics.autocorrelation_summary(posteriors = posteriors)

        intercept  YearsExperience  variance
Lag 0    1.000000         1.000000  1.000000
Lag 1    0.019390         0.039420  0.034288
Lag 5    0.013257        -0.018416  0.016291
Lag 10  -0.015301         0.009640 -0.031066
Lag 30   0.005944         0.015346 -0.007005
```

```python
gs.diagnostics.autocorrelation_plot(posteriors = posteriors)
```

### Posteriors Analysis

Asses posterior analysis:

```python
gs.analysis.trace_plot(posteriors = posteriors)
```

```python
gs.analysis.residuals_plot(posteriors = posteriors, data = data, response_variable = 'y')
```

```python
gs.analysis.summary(posteriors = posteriors)

Number of chains:           3
Sample size per chian:    500

Empirical mean, standard deviation, 95% HPD interval for each variable:

                         Mean            SD       HPD min       HPD max
intercept        2.502233e+04  2.344689e+03  2.053892e+04  2.965342e+04
YearsExperience  9.421604e+03  3.851400e+02  8.671903e+03  1.012988e+04
variance         3.454205e+07  9.457282e+06  1.934886e+07  5.389312e+07

Quantiles for each variable:

                         2.5%           25%           50%           75%         97.5%
intercept        2.026500e+04  2.354437e+04  2.494174e+04  2.660592e+04  2.945207e+04
YearsExperience  8.683961e+03  9.160491e+03  9.429675e+03  9.664610e+03  1.014999e+04
variance         2.094755e+07  2.792063e+07  3.275464e+07  3.925156e+07  5.773319e+07
```

```python
gs.analysis.compute_DIC(posteriors = posteriors, data = data, response_variable = 'y')

Deviance at posterior means           548.70
Posterior mean deviance               547.03
Effective number of parameteres        -1.67
Deviace Information Criterion         545.36
```

Predict the `Salary` distribution for a predictor `YearsExperience = 5`:

```python
distribution = gs.analysis.predict_distribution(posteriors = posteriors, predictors = {'YearsExperience': 5})

fig_2, ax_2 = plt.subplots()

ax_2.hist(distribution, bins = int(np.sqrt(len(distribution))), color = 'blue', alpha = 0.5, density = True)

ax_2.set_xlabel('Salary')
ax_2.set_ylabel('Probability Density')
ax_2.tick_params(bottom = False, top = False, left = False, right = False)

plt.tight_layout()

plt.show()
```

Comparing data to fitted model posteriors:

```python
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
```
