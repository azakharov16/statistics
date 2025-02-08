import numpy as np
import scipy.stats as ss
from scipy.linalg import cholesky
from pathlib import Path
from spot_portfolio import PassiveSpotPortfolio
import pickle

path = Path(r'/home/andrey/TRADING/crypto/bybit_data')

x_arr = np.random.standard_normal(size=1000)
z_arr = np.random.standard_normal(size=1000)
rho = 0.2
y_arr = rho * x_arr + np.sqrt(1 - rho ** 2) * z_arr
x_prob = ss.norm.cdf(x_arr)
y_prob = ss.norm.cdf(y_arr)
xx_arr = ss.t.ppf(x_prob, df=10)
yy_arr = ss.t.ppf(y_prob, df=5)
np.corrcoef(xx_arr, yy_arr)

with open(path.parent.joinpath('portfolios/portfolio_MARKOWITZ_bd41eac6-cdc1-4612-8869-0c4d21af46f5.pkl'), 'rb') as handle:
    port = pickle.load(handle)
    handle.close()

corr_mat = np.array(port.asset_stats['hist_corr'])
cov_mat = np.array(port.asset_stats['hist_cov'])
L_mat = cholesky(cov_mat, lower=True)
np.allclose(np.matmul(L_mat, L_mat.T), cov_mat)  # True
Z_mat = np.random.standard_normal(size=(23, 10000))
X_mat = np.matmul(L_mat, Z_mat)
corr_chol = np.corrcoef(X_mat, rowvar=True)
np.allclose(corr_chol, corr_mat, rtol=10e-2)
