"""
Two-sample tests to determine if two histograms are "equal".

"""
from __future__ import division

import numpy as np
import scipy.stats as stats

__all__ = [
    'chisq_twosample',
    'bayesian_twosample',
    'bv_nonextreme_twosample',
]

def chisq_twosample(countsX, countsY, alpha):
    """
    Calculates the chi-squared two-sample test between `countsX` and `countsY`.

    The goal is to determine if we should reject the null hypothesis which says
    that `countsX` and `countsY` were generated from the same distribution.

    Parameters
    ----------
    countsX : array-like
        The counts for X.
    countsY : array-like
        The counts for Y.
    alpha : float
        The significance level used to determine if the null hypothesis should
        be rejected.

    Returns
    -------
    reject : bool
        If `True`, then the null hypothesis is rejected at significance level
        `alpha`, and so the counts should be considered as generated from
        different distributions.
    pvalue : float
        The calculated p-value.

    Examples
    --------
    >>> chisq_twosample([1,10], [2,3], .05)
    (False, 0.34032457722408593)
    >>> chisq_twosample([1,30], [20,30], .05)
    (True, 0.001185539179966888)

    References
    ----------
    .. [1] http://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm

    """
    countsX = np.asarray(countsX)
    countsY = np.asarray(countsY)
    Xsum = countsX.sum()
    Ysum = countsY.sum()

    # The dof is reduced by 1 if the sample sizes are the same.
    if Xsum == Ysum:
        c = 1
    else:
        c = 0

    rescaleX = np.sqrt(Ysum / Xsum)
    rescaleY = 1 / rescaleX
    chisq = (rescaleX * countsX - rescaleY * countsY)**2 / (countsX + countsY)

    # Note that if there are any bins where the counts are simultaneously zero,
    # then the element in chisq is NaN. So we must do nansum to obtain the
    # value of chisq. Also, we can use the number of NaNs to update the dof.

    # The degrees of freedom for the chi-squared percent point function.
    k = len(countsY) - np.isnan(chisq).sum()
    dof = k - c

    chisq = np.nansum(chisq)
    reject = chisq > stats.chi2.ppf(1 - alpha, dof)
    pvalue = 1 - stats.chi2.cdf(chisq, dof)

    return reject, pvalue


def bayesian_twosample(countsX, countsY, prior=None):
    """
    Calculates a Bayesian-like two-sample test between `countsX` and `countsY`.

    The idea is taken from [1]_. We assume the counts are generated IID. Then
    we use Dirichlet prior to infer the underlying discrete distribution.

    In the null hypothesis, we say that `countsX` and `countsY` were generated
    from the same underlying distribution q. The alternative hypothesis is that
    `countsX` and `countsY` were generated by different distribution.

    A log Bayes factor is calculated:

        \chi = log P(X, Y | H_1) - log P(X, Y | H_0)

    If \chi is greater than 0, then reject the null hypothesis.

    To calculate P(X, Y | H_1), we calculate the product of evidences from
    two independent Bayesian inferences. That is, P(X)P(Y). To calculate
    P(X, Y | H_0), we combine the counts and calculate the evidence from a
    single Bayesian inference.

    Parameters
    ----------
    countsX : array-like, shape (n,)
        The counts for X.
    countsY : array-like, shape (n,)
        The counts for Y.
    prior : array-like, shape (n,)
        The Dirichlet hyper-parameters to use during inference. If `None`, we
        use Jeffrey's prior.

    Returns
    -------
    reject : bool
        If `True`, then the null hypothesis is rejected and the counts should
        be considered as generated from different distributions.
    chi : float
        The base-2 logarithm of the evidence ratio. If this value is greater
        than 0, then we reject the null hypothesis.

    Examples
    --------
    >>> bayesian_twosample([1,10], [2,3])
    (True, 0.11798407303051839)
    >>> bayesian_twosample([1,30], [20,30])
    (True, 9.4347501426274931)

    References
    ----------
    .. [1] Karsten M. Borgwardt and Zoubin Ghahramani, "Bayesian two-sample
           tests". http://arxiv.org/abs/0906.4032

    """
    if prior is None:
        # Use Jeffrey's prior for Dirichlet distributions.
        prior = np.ones(len(countsX)) * 0.5

    countsX = np.asarray(countsX)
    countsY = np.asarray(countsY)
    chi = log_evidence(countsX, prior) + log_evidence(countsY, prior)
    chi -= log_evidence(countsX + countsY, prior)
    reject = chi > 0
    return reject, chi


def log_evidence(counts, prior):
    """
    Returns the base-2 log evidence from counts using a Dirichlet prior.

    """
    from scipy.special import gammaln
    from math import log

    counts = np.asarray(counts)
    prior = np.asarray(prior)
    both = prior + counts

    evid = gammaln(both).sum() - gammaln(both.sum()) \
         + gammaln(prior.sum()) - gammaln(prior).sum()

    return evid / log(2)

def bv_nonextreme_twosample(countsX, countsY, tol):
    """
    Returns the nonextreme Z statistic of Bhattacharya and Valiant [1]_.

    Parameters
    ----------
    countsX : array-like
        The counts for X.
    countsY : array-like
        The counts for Y.
    tol : float
        The tolerance level for determining when to reject the null hypothesis.
        Choosing a value for the tolerance can be difficult, but generally
        values of Z greater than 0.1 or 0.2 seem to correlate with the
        chisquared two-sample and Bayesian two-sample tests give.

    Returns
    -------
    reject : bool
        If `True`, then the null hypothesis is rejected if the calculate Z
        value is greater than tol, and so the counts should be considered as
        generated from different distributions.
    Z : float
        The calculated Z-statistic.

    Examples
    --------
    >>> bv_nonextreme_twosample([1,10], [2,3], .05)
    (False, -0.096427404373121611)
    >>> bv_nonextreme_twosample([1,30], [20,30], .05)
    (True, 2.0998342829538652)

    References
    ----------
    .. [1] http://arxiv.org/abs/1504.04599

    """
    countsX = np.asarray(countsX)
    countsY = np.asarray(countsY)

    m1 = countsX.sum()
    m2 = countsY.sum()
    num = (m2 * countsX- m1 * countsY)**2 - (m2**2 * countsX + m1**2 * countsY)
    den = m1**(3/2) * m2 * (countsX + countsY)
    Z = (num / den).sum()
    reject = Z > tol
    return reject, Z
