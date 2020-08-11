import math

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class TobitRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, p_censor_left, p_censor_right, C, alpha, verbose=False):
        self._p_censor_left = p_censor_left
        self._p_censor_right = p_censor_right
        self._C = C
        self._alpha = alpha
        self._verbose = verbose
        self.ols_coef_ = None
        self.ols_intercept = None
        self.coef_ = None
        self.intercept_ = None
        self.sigma_ = None

    def fit(self, X, y):
        """
        Fit a maximum-likelihood Tobit regression
        :param X: Pandas DataFrame (n_samples, n_features): Data
        :param y: Pandas Series (n_samples,): Target
        :return:
        """
        X = X.copy()
        y = y.copy()

        # Remove index to avoid incorrect numpy broadcast
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        # Initialize the guess with OLS
        X.insert(0, 'intercept', 1.0)
        init_reg = LinearRegression(fit_intercept=False).fit(X, y)
        b0 = init_reg.coef_
        y_pred = init_reg.predict(X)
        resid = y - y_pred
        resid_var = np.var(resid)
        s0 = np.sqrt(resid_var)
        params0 = np.append(b0, s0)

        # L-BFGS-B optimization
        result = scipy.optimize.minimize(lambda params: self._tobit_neg_log_likelihood(X, y, params), params0, method='L-BFGS-B',
                          jac=lambda params: self._tobit_neg_log_likelihood_der(X, y, params), options={'disp': self._verbose})
        if self._verbose:
            print(result)

        # Construct the result
        self.ols_coef_ = b0[1:]
        self.ols_intercept = b0[0]
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:-1]
        self.sigma_ = result.x[-1]
        return self

    def predict(self, x):
        return self.intercept_ + x @ self.coef_

    def _get_censor_idx(self, y):
        idx_left = pd.Series(y) <= self._p_censor_left
        idx_mid = (pd.Series(y) > self._p_censor_left) & (pd.Series(y) < self._p_censor_right)
        idx_right = pd.Series(y) >= self._p_censor_right
        return idx_left.tolist(), idx_mid.tolist(), idx_right.tolist()

    def _tobit_neg_log_likelihood(self, xs, ys, params):
        idx_left, idx_mid, idx_right = self._get_censor_idx(ys)
        x_left, x_mid, x_right = xs[idx_left], xs[idx_mid], xs[idx_right]
        y_left, y_mid, y_right = ys[idx_left], ys[idx_mid], ys[idx_right]

        for df in [x_left, x_mid, x_right, y_left, y_mid, y_right]:
            df.reset_index(drop=True, inplace=True)

        b = params[:-1]
        s = params[-1]

        to_cat = []

        cens = False
        if len(idx_left) > 0:
            cens = True
            left = (y_left - x_left@b)
            to_cat.append(left)
        if len(idx_right) > 0:
            cens = True
            right = (x_right@b - y_right)
            to_cat.append(right)
        if cens:
            concat_stats = np.concatenate(to_cat, axis=0) / s
            log_cum_norm = scipy.stats.norm.logcdf(concat_stats)  # log_ndtr(concat_stats)
            cens_sum = log_cum_norm.sum()
        else:
            cens_sum = 0

        if len(idx_mid) > 0:
            mid_stats = (y_mid - x_mid@b) / s
            mid = scipy.stats.norm.logpdf(mid_stats) - math.log(max(np.finfo('float').resolution, s))
            mid_sum = mid.sum()
        else:
            mid_sum = 0

        loglik = cens_sum + mid_sum

        l1 = scipy.linalg.norm(params, ord=1)
        l2 = scipy.linalg.norm(params, ord=2) ** 2 / 2
        loglik -= self._C * (self._alpha * l1 + (1 - self._alpha) * l2)

        return -loglik

    def _tobit_neg_log_likelihood_der(self, xs, ys, params):
        idx_left, idx_mid, idx_right = self._get_censor_idx(ys)
        x_left, x_mid, x_right = xs[idx_left], xs[idx_mid], xs[idx_right]
        y_left, y_mid, y_right = ys[idx_left], ys[idx_mid], ys[idx_right]

        for df in [x_left, x_mid, x_right, y_left, y_mid, y_right]:
            df.reset_index(drop=True, inplace=True)

        b = params[:-1]
        # s = math.exp(params[-1]) # in censReg, not using chain rule as below; they optimize in terms of log(s)
        s = params[-1]

        beta_jac = np.zeros(len(b))
        sigma_jac = 0

        if len(idx_left) > 0:
            left_stats = (y_left - x_left@b) / s
            l_pdf = scipy.stats.norm.logpdf(left_stats)
            l_cdf = scipy.special.log_ndtr(left_stats)
            left_frac = np.exp(l_pdf - l_cdf)
            beta_left = left_frac@(x_left / s)
            beta_jac -= beta_left

            left_sigma = left_frac@left_stats
            sigma_jac -= left_sigma

        if len(idx_right) > 0:
            right_stats = (x_right@b - y_right) / s
            r_pdf = scipy.stats.norm.logpdf(right_stats)
            r_cdf = scipy.special.log_ndtr(right_stats)
            right_frac = np.exp(r_pdf - r_cdf)
            beta_right = right_frac@(x_right / s)
            beta_jac += beta_right

            right_sigma = right_frac@right_stats
            sigma_jac -= right_sigma

        if len(idx_mid) > 0:
            mid_stats = (y_mid - x_mid@b) / s
            beta_mid = mid_stats@(x_mid / s)
            beta_jac += beta_mid

            mid_sigma = (np.square(mid_stats) - 1).sum()
            sigma_jac += mid_sigma

        combo_jac = np.append(beta_jac, sigma_jac / s)  # by chain rule, since the expression above is dloglik/dlogsigma

        l1 = np.vectorize(lambda e: 1 if e > 0 else -1 if e < 0 else 0)(params)
        l2 = params
        combo_jac -= self._C * (self._alpha * l1 + (1 - self._alpha) * l2)

        return -combo_jac

    def score(self, X, y, sample_weight=None):
        return -self._tobit_neg_log_likelihood(X, y, [*self.coef_, self.sigma_])

    def get_params(self, deep=True):
        return {
            "C": self._C,
            "alpha": self._alpha,
            "p_censor_left": self._p_censor_left,
            "p_censor_right": self._p_censor_right,
            "verbose": self._verbose,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
