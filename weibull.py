from functools import partial
from itertools import combinations
import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy.special import gamma
from scipy.optimize import fsolve
from matplotlib import pyplot as plt

path = ""


def weibull_cdf(t, k, lamb):
    prob = 1 - np.exp(-(t / lamb) ** k)
    return prob


# Check
tenors = np.linspace(0.01, 30, num=10000)
scale = 1.5  # lambda
probs = weibull_cdf(tenors, k=1.2, lamb=scale)
probs_ss = ss.weibull_min.cdf(tenors / scale, c=1.2)
sp.allclose(probs, probs_ss)  # True

for lamb in np.arange(0.5, 1.6, 0.5):
    for k in np.arange(0.5, 1.6, 0.2):
        weibull_cdf_vec = np.vectorize(partial(weibull_cdf, k=k, lamb=lamb))
        plt.plot(tenors, weibull_cdf_vec(tenors), label=f"k={k:.1f}")
    plt.legend()
    plt.title(f"Weibull cumulative PD plots for lambda={lamb:.1f}")
    plt.savefig(path + f"weibull_cdf_plots_lambda={lamb:.1f}.png")


def weibull_hazard(t, k, lamb):
    return (k / lamb) * (t / lamb) ** (k - 1)


def weibull_hr(t, k_i, k_j, lamb_i, lamb_j):
    return (k_i * lamb_j ** k_j / k_j * lamb_i ** k_i) * t ** (k_i - k_j)


k_range = np.arange(0.5, 1.6, 0.2).tolist()
combn = list(combinations(k_range, 2))
for lamb in np.arange(0.5, 1.6, 0.5):
    for tup in combn:
        weibull_hr_vec = np.vectorize(partial(weibull_hr, k_i=tup[0], k_j=tup[1], lamb_i=lamb, lamb_j=lamb))
        plt.plot(tenors, weibull_hr_vec(tenors), label=f"HR for ki/kj={tup[0]:.1f}/{tup[1]:.1f}")
    plt.legend()
    plt.title(f"Hazard ratios for lambda={lamb:.1f}")
    plt.savefig(path + f"weibull_hr_plots_lambda={lamb:.1f}.png")


def series_variation(x, maxlag, norm='L2'):
    x_initial = x[maxlag:]
    x_shifted = x[:len(x) - maxlag]
    if norm == 'L2':
        x_diff = (x_initial - x_shifted) ** 2
    elif norm == 'L1':
        x_diff = np.abs(x_initial - x_shifted)
    else:
        raise ValueError("The norm must be 'L1' or 'L2'")
    return np.nanmax(x_diff)


weibull_hr_vec = np.vectorize(partial(weibull_hr, k_i=0.5, k_j=1.5, lamb_i=1.0, lamb_j=1.0))
series_variation(weibull_hr_vec(tenors[1000:]), maxlag=10, norm='L1')
mean_hr = np.mean(weibull_hr_vec(tenors[1000:]))


def shrinking_diff(x):
    mu = np.nanmean(x)
    mu_vec = []
    for i in range(len(x)):
        x_diff = np.abs(x[i:] - mu)
        mu_vec.append(np.nanmean(x_diff))
    return np.array(mu_vec)


m_shrink = shrinking_diff(weibull_hr_vec(tenors[1000:]))
plt.plot(tenors[1000:], m_shrink)
plt.show()


def rolling_diff(x, maxlag):
    mu = np.nanmean(x)
    mu_vec = []
    for i in range(len(x) - maxlag):
        x_diff = np.abs(x[i:i + maxlag] - mu)
        mu_vec.append(np.nanmean(x_diff))
    return np.array(mu_vec)


m_roll = rolling_diff(weibull_hr_vec(tenors[1000:]), maxlag=100)
plt.plot(tenors[1100:], m_roll)
plt.show()


def weibull_marginal(t1_vec, t2_vec, k, lamb):
    weibull_cdf_vec = np.vectorize(partial(weibull_cdf, k=k, lamb=lamb))
    PD_cumm1 = weibull_cdf_vec(t1_vec)
    PD_cumm2 = weibull_cdf_vec(t2_vec)
    return PD_cumm2 - PD_cumm1


plt.plot(tenors, weibull_marginal(tenors, tenors + 1.0, k=0.8, lamb=100))  # decreasing
plt.plot(tenors, weibull_marginal(tenors, tenors + 1.0, k=1.3, lamb=100))  # increasing
# However, for lambdas of scale 1x or 10x and for k close to 1 the relationship may be non-monotone
plt.plot(tenors, weibull_marginal(tenors, tenors + 1.0, k=1.01, lamb=100))


# Maximum likelihood estimation of Weibull parameters
surv_times = np.random.exponential(120, size=5000)  # k = 1, lambda = 120
surv_times_high = np.random.weibull(1.2, size=5000)
surv_times_low = np.random.weibull(0.8, size=5000)


def WeibullParams_MLE(sample):
    def mlestim_k(x_vec):
        def obj_func(k):
            return np.sum(x_vec ** k * np.log(x_vec)) / np.sum(x_vec ** k) - 1 / k - np.mean(np.log(x_vec))
        res = fsolve(obj_func, x0=np.array(1.0))
        return res[0]

    def mlestim_lambda(k, x_vec):
        return (np.mean(x_vec ** k)) ** (1 / k)

    k_hat = mlestim_k(x_vec=sample)
    lambda_hat = mlestim_lambda(k_hat, x_vec=sample)
    return k_hat, lambda_hat


k_MLE, lambda_MLE = WeibullParams_MLE(surv_times)
k_MLE_high, lambda_MLE_high = WeibullParams_MLE(surv_times_high)  # data is scaled, lambda = 1
k_MLE_low, lambda_MLE_low = WeibullParams_MLE(surv_times_low)


# Method of moments estimation of Weibull parameters
def WeibullParams_MM(sample):
    def mmestim_k(x_vec):
        mu = np.mean(x_vec)
        var = np.var(x_vec, ddof=1)
        def obj_func(k):
            return gamma(1 + 2 / k) / (gamma(1 + 1 / k) ** 2) - 1 - var / mu ** 2
        res = fsolve(obj_func, x0=np.array(1.0))
        return res[0]

    def mmestim_lambda(k, x_vec):
        mu = np.mean(x_vec)
        return mu / gamma(1 + 1 / k)
    k_hat = mmestim_k(x_vec=sample)
    lambda_hat = mmestim_lambda(k_hat, x_vec=sample)
    return k_hat, lambda_hat


k_MM, lambda_MM = WeibullParams_MM(surv_times)
k_MM_high, lambda_MM_high = WeibullParams_MM(surv_times_high)
k_MM_low, lambda_MM_low = WeibullParams_MM(surv_times_low)


# Estimating Weibull parameters by OLS
def WeibullFit(time_vec, dr_vec):
    y_vec = np.log(-np.log(1 - dr_vec))
    n = len(y_vec)
    X_mat = np.concatenate((np.ones(n)[:, np.newaxis], time_vec[:, np.newaxis]), axis=1)
    beta_vec = np.dot(np.linalg.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, y_vec))
    lambda_hat = np.exp(-beta_vec[0] / beta_vec[1])
    k_hat = beta_vec[1]
    return k_hat, lambda_hat

