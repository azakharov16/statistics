import numpy as np
import pandas as pd
import scipy.stats as ss
import plotly.express as px
import plotly.io as pio
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import minimize
from tqdm import tqdm

pio.renderers.default = 'browser'

def hansen_vpdf(x, n, ksi):
    c = gamma((n + 1) / 2) / (gamma(n / 2) * np.sqrt(np.pi * (n - 2)))
    a = 4.0 * ksi * c * (n - 2) / (n - 1)
    b = np.sqrt(1.0 + 3.0 * ksi ** 2 - a ** 2)
    d = np.where(x < (- a / b), -1, 1)
    denom = 1.0 + d * ksi
    res = b * c * (1.0 + (((b * x + a) / denom) ** 2) / (n - 2)) ** (-(n + 1) / 2)
    return res

def hansen_pdf(x, n, ksi):
    c = gamma((n + 1) / 2) / (gamma(n / 2) * np.sqrt(np.pi * (n - 2)))
    a = 4.0 * ksi * c * (n - 2) / (n - 1)
    b = np.sqrt(1.0 + 3.0 * ksi ** 2 - a ** 2)
    if x < (- a / b):
        d = 1.0 - ksi
    else:
        d = 1.0 + ksi
    res = b * c * (1.0 + (((b * x + a) / d) ** 2) / (n - 2)) ** (-(n + 1) / 2)
    return res

def hansen_cdf(x, n, ksi):
    res = quad(hansen_pdf, -np.inf, x, args=(n, ksi))
    return res[0]

def hansen_ppf(alpha, n, ksi):
    def obj_func(x):
        return (alpha - hansen_cdf(x, n, ksi)) ** 2
    res = minimize(obj_func, x0=np.array([0.0]), method='BFGS', tol=10e-8)
    return res.x[0]

x_vec = np.linspace(-10, 10, 1000)
y_vec = []
for x in x_vec:
    y_vec.append(hansen_cdf(x, n=10, ksi=-0.5))

df1 = pd.DataFrame({'X': x_vec, 'PDF': hansen_vpdf(x_vec, n=10, ksi=-0.5), 'PDF_NORM': ss.norm.pdf(x_vec)})
fig = px.line(df1, x='X', y=df1.drop('X', axis=1).columns)
fig.update_layout(template='plotly_dark', autosize=False, height=800, width=1815)
fig.show()

df2 = pd.DataFrame({'X': x_vec, 'CDF': np.array(y_vec)})
fig = px.line(df2, x='X', y='CDF')
fig.update_layout(template='plotly_dark', autosize=False, height=800, width=1815)
fig.show()

q = hansen_ppf(0.05, n=10, ksi=-0.5)
print(hansen_cdf(q, n=10, ksi=-0.5))

# Generate RVS of Hansen-t (left tail)
z_sim = np.random.standard_normal(size=10000)
prob_z = ss.norm.cdf(z_sim)
x_sim = []
for p in tqdm(prob_z):
    x_sim.append(hansen_ppf(p, n=10, ksi=-0.5))
x_sim = np.array(x_sim)
x_tail = x_sim[x_sim < np.quantile(x_sim, q=0.1)]
print(ss.genpareto.fit(data=np.abs(x_tail), method='MLE'))
