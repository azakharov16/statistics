import warnings
import numpy as np
import scipy.stats as ss
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.stattools as st

# Testing: simulate autocorrelated series y as y(t) = rho * y(t-1) + X(t)*b + e(t)
# Other ideas: Durbin's h, Wald, White, Chow, Goldfeld-Quandt, Breusch-Pagan, F-test, F-test with restrictions, DWH test
# White SEs, Breusch-Godfrey, Newey-West SEs, DM test, MZ regressions, DF, ADF, KPSS, PP, F-test for dummies, Hausman,
# LR test, Wilcoxon signed-rank test, Mann-Whitney U-test, Theil U-test, Shapiro-Wilk test


def JarqueBera(y_vec, y_hat, nreg=1, alpha=0.05):
    resids = y_vec - y_hat
    nobs = resids.size
    sk = ss.skew(resids)
    kurt = ss.kurtosis(resids, fisher=False)
    JB = (nobs - nreg + 1) * ((sk ** 2) / 6 + ((kurt - 3) ** 2) / 24)
    if JB > ss.chi2.ppf(1 - alpha, 2):
        # The H0 hypothesis of sk=0 and kurt=3 is rejected
        result = False
    else:
        result = True
    pval = 1 - ss.chi2.cdf(JB, 2)
    return JB, pval, result


def create_test_series(nobs, nregressors, rho=0.0, betas=None):
    u_vec = np.zeros(nobs)
    X_mat = np.random.standard_normal(size=(nregressors, nobs))
    if betas is None:
        beta_vec = np.random.uniform(-1, 2, size=nregressors)
    else:
        beta_vec = betas
    e_vec = np.random.standard_normal(size=nobs)
    u_vec[0] = e_vec[0]
    for i in range(1, nobs):
        u_vec[i] = rho * u_vec[i - 1] + e_vec[i]
    Xb_vec = np.dot(beta_vec[np.newaxis, :], X_mat).flatten()
    y_vec = Xb_vec + u_vec
    return X_mat, y_vec


class DWCriticalValue(object):
    def __init__(self, alpha, filename):
        self.alpha = alpha
        self.tabs = self.__load(filename=filename)
        if self.alpha == 0.05:
            self.lookup_tab = self.tabs['level_95']
        elif self.alpha == 0.01:
            self.lookup_tab = self.tabs['level_99']
        else:
            raise LookupError("Only 5% and 1% confidence levels are currently supported")

    @staticmethod
    def __load(filename):
        store = pd.HDFStore(filename, 'r')
        tabs = {'level_95': store['level_95'], 'level_99': store['level_99']}
        store.close()
        return tabs

    def lookup(self, n, k, kind):
        if (n - k) < 5 or k > 20:
            raise LookupError("The critical values are unavailable for these values of n and k")
        if kind not in ('upper', 'lower'):
            raise ValueError("Wrong DW statistic type: must be either 'lower' or 'upper'")
        col = {'upper': 'dU', 'lower': 'dL'}[kind]
        if n not in self.lookup_tab.index.values:
            warnings.warn("The sample size is not in the table, critical value will be interpolated")
            self.lookup_tab = self.lookup_tab[self.lookup_tab[(k, col)].notna()]
            dw_crit = np.interp(n, self.lookup_tab.index.values.tolist(), self.lookup_tab[(k, col)].values.tolist())
        else:
            dw_crit = self.lookup_tab.at[n, (k, col)]
        return dw_crit


def DWtest(X_mat, y_vec, beta_vec, alpha, dw_tabs='DW_crit_values.hd5'):
    resid_vec = y_vec - np.dot(beta_vec[np.newaxis, :], X_mat).flatten()
    DW = 2 - 2 * np.sum(resid_vec[:-1] * resid_vec[1:]) / np.sum(resid_vec ** 2)
    crit_val = DWCriticalValue(alpha, dw_tabs)
    dl = crit_val.lookup(y_vec.shape[0], X_mat.shape[0], 'lower')
    du = crit_val.lookup(y_vec.shape[0], X_mat.shape[0], 'upper')
    if DW < dl:
        # Hypothesis of independence is rejected, autocorrelation is present
        result = False
    elif DW > du:
        result = True
    else:
        warnings.warn("The DW test was inconclusive")
        result = None
    return DW, result


X_sim, y_sim = create_test_series(110, 4, rho=0.5)
lm = sm.OLS(y_sim, X_sim.transpose())
reg_result = lm.fit()
print(reg_result.summary())
#with open('reglog_testing.txt', 'w') as f:
#    print(reg_result.summary(), file=f)
#    f.close()
print(reg_result.aic, reg_result.bic, reg_result.nobs)
print(reg_result.rsquared, reg_result.rsquared_adj)
print(reg_result.fvalue, reg_result.f_pvalue)
y_hat = reg_result.fittedvalues
beta_hat = reg_result.params
resids = reg_result.resid
np.array_equal(y_sim - y_hat, reg_result.resid)  # True
t_stats = (reg_result.tvalues, reg_result.pvalues)
print(reg_result.ess + reg_result.ssr == reg_result.uncentered_tss)  # True

JB_sm, pval_sm, *rest = st.jarque_bera(resids)
JB_ss, pval_ss = ss.jarque_bera(resids)
JB_my, pval_my, res = JarqueBera(y_sim, y_hat, 1, 0.05)
DW = st.durbin_watson(resids)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    DW_my = DWtest(X_sim, y_sim, beta_hat, 0.05)  # autocorrelation detected

