from functools import partial
from itertools import combinations
from timeit import timeit
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.anova import anova_lm

### Chi-square criterion ###

data = np.random.standard_normal(size=1000)
freq_observed, edges = np.histogram(data, bins=50, density=False)
freq_observed = freq_observed / np.sum(freq_observed)
freq_expected = ss.norm.cdf(edges[1:]) - ss.norm.cdf(edges[:-1])
chi2val_ss, pval_ss = ss.chisquare(freq_observed, freq_expected)  # not rejected with pvalue = 1.0


def ChiSquare(freqs, probs, alpha=0.05, normalized=False):
    if not normalized:
        freqs = freqs / np.sum(freqs)
    ngroups = len(freqs)
    chi2_stat = np.sum((freqs - probs) ** 2 / probs)
    chi2_crit = ss.chi2.ppf(1 - alpha, ngroups - 1)
    if chi2_stat < chi2_crit:
        # H0: the observed distribution is the same as theoretical - not rejected
        res = True
    else:
        res = False
    pval = 1 - ss.chi2.cdf(chi2_stat, ngroups - 1)
    return res, chi2_stat, pval


res, chi2_val_my, pval_my = ChiSquare(freq_observed, freq_expected)


### Multiple correlation coefficient ###

x = np.random.standard_normal(size=(5, 100))
corr_mat = np.corrcoef(x)  # by row


def tr(array, k=0):
    if array.shape[0] != array.shape[1]:
        raise ValueError("The matrix must be square")
    return np.diag(array, k).sum()


def get_minor(array, i, j):
    if len(array.shape) != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("The input array must be a square matrix")
    row = np.arange(0, array.shape[0])
    col = np.arange(0, array.shape[1])
    return array[np.delete(row, i)[:, np.newaxis], np.delete(col, j)]


def mult_corrcoef(i, corr_mat):
    alg_complement = get_minor(corr_mat, i, i)
    return np.sqrt(1 - np.linalg.det(corr_mat) / np.linalg.det(alg_complement))


mult_corr_part = partial(mult_corrcoef, corr_mat=corr_mat)
mult_corr_vec = np.vectorize(mult_corr_part)
print(mult_corr_vec(np.arange(x.shape[0])))


### Variance Inflation Factor ###

x1 = np.random.standard_normal(size=1000)
rho12 = 0.8
x2 = np.sqrt(1 - rho12 ** 2) * x1 + rho12 * np.random.standard_normal(size=1000)
rho23 = 0.6
x3 = np.sqrt(1 - rho23 ** 2) * x2 + rho23 * np.random.standard_normal(size=1000)
X_mat = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis], x3[:, np.newaxis]), axis=1)

VIF1 = vif(X_mat, 0)
VIF2 = vif(X_mat, 1)
VIF3 = vif(X_mat, 2)


def compute_vif(X_mat, x_ind):
    if X_mat.shape[0] < X_mat.shape[1]:
        X_mat = np.transpose(X_mat)
    n = X_mat.shape[0]
    y = X_mat[:, x_ind].flatten()
    X_mat = np.concatenate((np.ones(n)[:, np.newaxis], np.delete(X_mat, x_ind, axis=1)), axis=1)
    alpha_vec = np.dot(np.linalg.inv(np.dot(X_mat.T, X_mat)), np.dot(X_mat.T, y))
    y_hat = np.dot(alpha_vec, X_mat.T)
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum((y - y_hat) ** 2)
    r2 = 1 - rss / tss
    vif = 1 / (1 - r2)
    return vif


VIF1_check = compute_vif(X_mat, 0)
VIF2_check = compute_vif(X_mat, 1)
VIF3_check = compute_vif(X_mat, 2)


def recombine_factors(X_mat, maxdeg=2):  # TODO allow all cross products of type (X1)^2 * X2
    if maxdeg < 2 or not isinstance(maxdeg, int):
        raise ValueError("The 'maxdeg' argument must be a positive integer")
    if X_mat.shape[0] < X_mat.shape[1]:
        X_mat = np.transpose(X_mat)
    nfactors = X_mat.shape[1]
    n = X_mat.shape[0]
    intercept_check = np.all(X_mat - np.ones(n)[:, np.newaxis], axis=0)
    if not np.all(intercept_check):
        ind = np.where(~intercept_check)[0].tolist().pop()
        X_mat = np.delete(X_mat, ind)
        nfactors -= 1
    X_mod = X_mat.copy()
    for d in range(2, maxdeg + 1):
        X_mod = np.append(X_mod, X_mat ** d, axis=1)
    for d in range(2, maxdeg + 1):
        comb = combinations(range(nfactors), d)
        for c in comb:
            X_cross = X_mat[:, c[0]][:, np.newaxis].copy()
            for k in range(1, len(c)):
                X_cross *= X_mat[:, c[k]][:, np.newaxis]
            X_mod = np.append(X_mod,  X_cross, axis=1)
    if not np.all(intercept_check):
        X_mod = np.append(np.ones(n)[:, np.newaxis], X_mod, axis=1)
    return X_mod


X = np.concatenate((2 * np.ones(shape=(1000, 1)), 3 * np.ones(shape=(1000, 1)), 4 * np.ones(shape=(1000, 1))), axis=1)
timeit(stmt="X_new = recombine_factors(X, maxdeg=3)", number=10000, globals=globals())


def white_test(X_mat, y_vec, beta_vec, alpha):
    resid_vec = y_vec - np.dot(beta_vec[np.newaxis, :], X_mat).flatten()
    n = len(resid_vec)
    resid_sq_vec = resid_vec ** 2
    X_aux_mat = recombine_factors(X_mat, maxdeg=2)
    gamma_vec = np.dot(np.linalg.inv(np.dot(X_aux_mat.T, X_aux_mat)), np.dot(X_aux_mat.T, resid_sq_vec))
    aux_resid_vec = resid_sq_vec - np.dot(gamma_vec[np.newaxis, :], X_aux_mat).flatten()
    nvars = X_aux_mat.shape[1]
    chi2_stat = n * np.sum(aux_resid_vec ** 2) / np.sum((resid_sq_vec - np.mean(resid_sq_vec)) ** 2)
    chi2_crit = ss.chi2.ppf(1 - alpha, nvars - 1)
    if chi2_stat < chi2_crit:
        # H0: homoscedasticity
        res = True
    else:
        res = False
    pval = 1 - ss.chi2.cdf(chi2_stat, nvars - 1)
    return res, chi2_stat, pval


# Conduct experiment!!!
# statsmodels.stats.diagnostic.het_white(resid, exog)


def create_het_sample(nobs, nregressors, betas=None):  # TODO correct dimensions
    X_mat = np.random.standard_normal(size=(nobs, nregressors))
    e_vec = np.random.standard_normal(size=nregressors)
    u_vec = np.dot(X_mat, e_vec[:, np.newaxis]).flatten()  # Hadamard product here!
    X_mat = np.append(np.ones(shape=(nobs, 1)), X_mat, axis=1)
    if betas is None:
        beta_vec = np.random.uniform(-1, 2, size=nregressors + 1)
    else:
        beta_vec = betas
    Xb_vec = np.dot(X_mat, beta_vec[:, np.newaxis]).flatten()
    y_vec = Xb_vec + u_vec
    return X_mat, y_vec


X_sim, y_sim = create_het_sample(1000, 4, betas=np.array([-1, 2, 1.5, 0.5, -0.5]))
lm = sm.OLS(y_sim, X_sim)
reg_result = lm.fit()
resids = reg_result.resid
beta_hat = reg_result.params
white_sm = het_white(resids, X_sim)  # TODO read about associated F-test
white_my = white_test(X_sim.T, y_sim, beta_hat, 0.05)


def anova_1way(y_name, x_name, data):
    df = data[[x_name, y_name]].copy()
    glob_mean = df[y_name].mean()
    ss_total = np.sum((df[y_name] - glob_mean) ** 2)
    group_means = df.groupby(x_name).mean().rename(columns={y_name: 'group_mean'})
    df = pd.merge(df, group_means, left_on=x_name, right_index=True)
    ss_resid = np.sum((df[y_name] - df['group_mean']) ** 2)
    ss_explained = np.sum((df['group_mean'] - glob_mean) ** 2)
    assert np.allclose(ss_total, ss_resid + ss_explained)
    n_groups = df[x_name].nunique()
    df1 = n_groups - 1
    df2 = df.shape[0] - n_groups
    ms_explained = ss_explained / df1
    ms_resid = ss_resid / df2
    f_stat = ms_explained / ms_resid
    pval = 1 - ss.f.cdf(f_stat, df1, df2)
    return f_stat, pval


y_sim = np.concatenate([
    np.random.normal(0.25, 1.0, size=1000),
    np.random.normal(0.5, 1.0, size=500),
    np.random.normal(0.75, 1.0, size=1500)])
x_sim = np.concatenate([np.repeat(1.0, 1000), np.repeat(2.0, 500), np.repeat(3.0, 1500)])
df_test = pd.DataFrame(columns=['y', 'categ'])
df_test['y'] = y_sim
df_test['categ'] = x_sim
f_my, pvalue_my = anova_1way('y', 'categ', data=df_test)
lm = smf.ols('y ~ categ', data=df_test).fit()
anova_tab = anova_lm(lm)
print(anova_tab)
f_sm, pvalue_sm = anova_tab['F'].loc['categ'], anova_tab['PR(>F)'].loc['categ']
