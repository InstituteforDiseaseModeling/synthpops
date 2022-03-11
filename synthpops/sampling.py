"""Sample distributions, either from real world data or from uniform distributions."""

import numpy as np
import numba as nb
import pandas as pd
import sciris as sc
import random
import itertools
import bisect
import scipy
from scipy import stats as st
import warnings
from . import base as spb
from . import plotting as sppl


def set_seed(seed=None):
    """Reset the random seed -- complicated because of Numba."""
    @nb.njit((nb.int64,), cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed)  # If None, reinitializes it
    if seed is None:  # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed)  # Finally, reset Python's built-in random number generator

    return


def fast_choice(weights):
    """
    Choose an option -- quickly -- from the provided weights. Weights do not need
    to be normalized.

    Reimplementation of random.choices(), removing everything inessential.

    Example:
        fast_choice([0.1,0.2,0.3,0.2,0.1]) # might return 2
    """
    cum_weights = list(itertools.accumulate(weights))
    return bisect.bisect(cum_weights, random.random()*(cum_weights[-1]), 0, len(cum_weights)-1)


# @nb.njit(cache=True)
def sample_single_dict(distr_keys, distr_vals):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.

    """
    return distr_keys[fast_choice(distr_vals)]


def sample_single_arr(distr):
    """
    Sample from a distribution.

    Args:
        distr (dict or np.ndarray): distribution

    Returns:
        A single sampled value from a distribution.
    """
    return fast_choice(distr)


def resample_age(age_dist_vals, age):
    """
    Resample age from single year age distribution.

    Args:
        single_year_age_distr (arr) : age distribution, ordered by age
        age (int)                   : age as an integer
    Returns:
        Resampled age as an integer.
    """
    if age == 0:
        age_min = 0
        age_max = 1
    elif age == 1:
        age_min = 0
        age_max = 2
    elif age >= 2 and age <= 98:
        age_min = age - 2
        age_max = age + 2
    elif age == 99:
        age_min = 97
        age_max = 99
    else:
        age_min = 98
        age_max = 100

    age_distr = age_dist_vals[age_min:age_max + 1]  # create an array of the values, not yet normalized
    age_range = np.arange(age_min, age_max + 1)
    return age_range[fast_choice(age_distr)]


def sample_from_range(distr, min_val, max_val):
    """
    Sample from a distribution from min_val to max_val, inclusive.

    Args:
        distr (dict)  : distribution with integer keys
        min_val (int) : minimum of the range to sample from
        max_val (int) : maximum of the range to sample from
    Returns:
        A sampled number from the range min_val to max_val in the distribution distr.
    """
    new_distr = spb.norm_age_group(distr, min_val, max_val)
    distr_keys = np.array(list(new_distr.keys()), dtype=np.int64)
    distr_vals = np.array(list(new_distr.values()), dtype=np.float64)
    return sample_single_dict(distr_keys, distr_vals)


def check_dist(actual, expected, std=None, dist='norm', check='dist', label=None, alpha=0.05, size=10000, verbose=True, die=False, stats=False):
    """
    Check whether counts match the expected distribution. The distribution can be
    any listed in scipy.stats. The parameters for the distribution should be supplied
    via the "expected" argument. The standard deviation for a normal distribution is
    a special case; it can be supplied separately or calculated from the (actual) data.

    Args:
        actual (int, float, or array) : the observed value, or distribution of values
        expected (int, float, tuple)  : the expected value; or, a tuple of arguments
        std (float)                   : for normal distributions, the standard deviation of the expected value (taken from data if not supplied)
        dist (str)                    : the type of distribution to use
        check (str)                   : what to check: 'dist' = entire distribution (default), 'mean' (equivalent to supplying np.mean(actual)), or 'median'
        label (str)                   : the name of the variable being tested
        alpha (float)                 : the significance level at which to reject the null hypothesis
        size (int)                    : the size of the sample from the expected distribution to compare with if distribution is discrete

        verbose (bool)                : print a warning if the null hypothesis is rejected
        die (bool)                    : raise an exception if the null hypothesis is rejected
        stats (bool)                  : whether to return statistics

    Returns:
        If stats is True, returns statistics: whether null hypothesis is
        rejected, pvalue, number of samples, expected quintiles, observed
        quintiles, and the observed quantile.

    **Examples**::

        sp.check_dist(actual=[3,4,4,2,3], expected=3, dist='poisson')
        sp.check_dist(actual=[0.14, -3.37,  0.59, -0.07], expected=0, std=1.0, dist='norm')
        sp.check_dist(actual=5.5, expected=(1, 5), dist='lognorm')
    """
    # Handle inputs
    label = f' "{label}"' if label else ''
    is_dist = sc.isiterable(actual)

    # Set distribution
    if dist.lower() in ['norm', 'normal', 'gaussian']:
        if std is None:
            if is_dist:
                std = np.std(actual)  # Get standard deviation from the data
            else: # pragma: no cover
                std = 1.0
        args = (expected, std)
        scipydist = getattr(scipy.stats, 'norm')
        truedist = scipy.stats.norm(expected, std)
    else:
        try:
            if sc.isnumber(expected):
                args = (expected, )
            else:
                args = tuple(expected)
            scipydist = getattr(scipy.stats, dist)
            truedist = scipydist(*args)
        except Exception as E:
            errormsg = f'Distribution "{dist}" not supported with the expected values supplied; valid distributions are those in scipy.stats'
            raise NotImplementedError(errormsg) from E

    # Calculate stats
    if is_dist and check == 'dist':
        quantile = truedist.cdf(np.median(actual))

        # only if distribution is continuous
        if isinstance(scipydist, scipy.stats.rv_continuous):
            teststat, pvalue = scipy.stats.kstest(rvs=actual, cdf=dist, args=args)  # Use the K-S test to see if came from the same distribution

        # ks test against large sample from the theoretical distribution
        elif isinstance(scipydist, scipy.stats.rv_discrete):
            expected_r = truedist.rvs(size=size)
            teststat, pvalue = scipy.stats.ks_2samp(actual, expected_r)

        else: # pragma: no cover
            errormsg = 'Distribution is neither continuous or discrete and so not supported at this time.'
            raise NotImplementedError(errormsg)
        null = pvalue > alpha

    else:
        if check == 'mean':
            value = np.mean(actual)
        elif check == 'median':
            value = np.median(actual)
        else:
            value = actual
        quantile = truedist.cdf(value)  # If it's a single value, see where it lands on the Poisson CDF
        pvalue = 1.0-2*abs(quantile-0.5)  # E.g., 0.975 maps on to p=0.05
        minquant = alpha/2  # e.g., 0.025 for alpha=0.05
        maxquant = 1-alpha/2  # e.g., 0.975 for alpha=0.05
        minval = truedist.ppf(minquant)
        maxval = truedist.ppf(maxquant)
        quant_check = (minquant <= quantile <= maxquant)  # True if above minimum and below maximum
        val_check = (minval <= value <= maxval)  # Check values
        null = quant_check or val_check  # Consider it to pass if either passes

    # Additional stats
    n_samples = len(actual) if is_dist else 1
    eps = 1.0/n_samples if n_samples > 4 else 1e-2  # For small number of samples, use default limits
    quintiles = [eps, 0.25, 0.5, 0.75, 1-eps]
    obvs_quin = np.quantile(actual, quintiles) if is_dist else actual
    expect_quin = truedist.ppf(quintiles)

    # If null hypothesis is rejected, print a warning or error
    if not null:
        msg = f''''
Variable{label} with n={n_samples} samples is out of range using the distribution:
    {dist}({args}) →
    p={pvalue} < α={alpha}
Expected quintiles are: {expect_quin}
Observed quintiles are: {obvs_quin}
Observed median is in quantile: {quantile}'''
        if die:
            raise ValueError(msg)
        elif verbose:
            warnings.warn(msg)

    # If null hypothesis is not rejected, under verbose, print a confirmation
    if null and verbose:
        print(f'Check passed. Null hypothesis with expected distribution: {dist}{args} not rejected.')
        if is_dist and check == 'dist':
            print(f'Test statistic: {teststat}, pvalue: {pvalue}')

    if not stats:
        return null
    else:
        s = sc.objdict()
        s.null = null
        s.pvalue = pvalue
        s.n_samples = n_samples
        s.expected_quintiles = expect_quin
        s.observed_quintiles = obvs_quin
        s.observed_quantile = quantile
        return s


def check_normal(*args, **kwargs):
    ''' Alias to check_dist(dist='normal') '''
    dist = kwargs.pop('dist', 'norm')
    return check_dist(*args, **kwargs, dist=dist)


def check_poisson(*args, **kwargs):
    ''' Alias to check_dist(dist='poisson') '''
    dist = kwargs.pop('dist', 'poisson')
    return check_dist(*args, **kwargs, dist=dist)


def check_truncated_poisson(testdata, mu, lowerbound=None, upperbound=None, skipcheck=False, **kwargs):
    """
    test if data fits in truncated poisson distribution between upperbound and lowerbound using kstest
    Args:
        testdata (array) : data to be tested
        mu (float) : expected mean for the poisson distribution
        lowerbound (float) : lowerbound for truncation
        upperbound (float) : upperbound for truncation

    Returns:
        (bool) return True if statistic check passed, else return False
    """
    sample_size = len(testdata)
    # need to exclude any value below or equal to lowerbound and any value above or equal to upperbound, so we first find the quantile location for
    # lowerbound and upperbound then only generate poisson cdf values in between these 2 locations
    minquantile = st.poisson.cdf(lowerbound, mu=mu) if lowerbound else 0
    maxquantile = st.poisson.cdf(upperbound, mu=mu) if upperbound else 1
    # create uniformly distributed number between minquantile and maxquantile (in the cdf quantile space)
    q = np.random.uniform(low=minquantile, high=maxquantile, size=sample_size)
    # use percent point function to get inverse of cdf
    expected_data = st.poisson.ppf(q=q, mu=mu)
    result = True
    if not skipcheck:
        try:
            statistic_test(expected_data, testdata, test=st.kstest, verbose=True, die=True)
            result = True
        except ValueError as e:
            if 'reject the hypothesis' in str(e):
                result = False
            else:
                raise Exception(e)

    #plot comparison
    bins_count = int(min(10, max(expected_data)-min(expected_data)+1))
    print(f"bins:{bins_count}")

    expected, bins = np.histogram(expected_data, bins=bins_count)
    actual = np.histogram(testdata, bins=bins)[0]
    kwargs["generated"] = actual
    #merge 11 bins to 10 for bar plot align at center
    merged_bins = [round((bins[idx] + bins[idx+1])/2,2) for idx, val in enumerate(bins) if idx < len(bins)-1]
    kwargs["xvalue"] = merged_bins
    if kwargs["do_show"]:
        sppl.plot_array(expected, **kwargs)
    return result


def statistic_test(expected, actual, test=st.chisquare, verbose=True, die=False, **kwargs):
    """
    Perform statistical checks for expected and actual data based on the null
    hypothesis that expected and actual distributions are identical. Throw
    assertion if the expected and actual data differ significantly based on the
    test selected.
    See https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests.

    Args:
        expected (array)    : the expected value; or, a tuple of arguments
        actual (array)      : the observed value, or distribution of values
        test (scipy.stats)  : scipy statistical tests functions, for example scipy.stats.chisquare
        verbose (bool)      : print a warning if the null hypothesis is rejected
        die (bool)          : raise an exception if the null hypothesis is rejected
        **kwargs (dict)     : optional arguments for statistical tests

    Returns:
        None.
    """
    # data = {'expected': expected, 'actual': actual}
    df_expected = pd.DataFrame(expected, columns=['expected'])
    df_actual = pd.DataFrame(actual, columns=['actual'])
    details = pd.merge(df_expected.describe(), df_actual.describe(), left_index=True, right_index=True, suffixes=('expected', 'actual'))

    print(f"use {str(test.__name__)} to check actual distribution")
    s, p = test(expected, actual, **kwargs)
    print(f"statistics: {s} pvalue:{p}")
    if p < 0.05: # pragma: no cover
        msg = f"Under the null hypothesis the expected/actual distributions are identical." \
              f"If statistics is small or the p-value is high (greater than the significance level 5%)," \
              f" then we cannot reject the hypothesis. But we got p = {p} and s = {s}."
        if die: # pragma: no cover
            raise ValueError(msg)
        elif verbose: # pragma: no cover
            warnings.warn(f"data: \n{details}")
            warnings.warn(msg)
