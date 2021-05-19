'''
Default values and mathematical utilities
'''

import numpy as np
import numba as nb

#%% Global settings
default_int   = np.int64
default_float = np.float64
nbint         = nb.int64
nbfloat       = nb.float64
cache         = True


__all__ = ['find_contacts', 'choose', 'choose_r', 'n_multinomial', 'poisson', 'n_poisson', 'n_neg_binomial']


@nb.njit((nbint[:], nbint[:], nb.int64[:]), cache=cache)
def find_contacts(p1, p2, inds): # pragma: no cover
    """
    Numba for Layer.find_contacts()

    A set is returned here rather than a sorted array so that custom tracing interventions can efficiently
    add extra people. For a version with sorting by default, see Layer.find_contacts(). Indices must be
    an int64 array since this is what's returned by true() etc. functions by default.
    """
    pairing_partners = set()
    inds = set(inds)
    for i in range(len(p1)):
        if p1[i] in inds:
            pairing_partners.add(p2[i])
        if p2[i] in inds:
            pairing_partners.add(p1[i])
    return pairing_partners


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose(max_n, n):
    '''
    Choose a subset of items (e.g., people) without replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose(5, 2) # choose 2 out of 5 people with equal probability (without repeats)
    '''
    return np.random.choice(max_n, n, replace=False)


@nb.njit((nbint, nbint), cache=cache) # Numba hugely increases performance
def choose_r(max_n, n):
    '''
    Choose a subset of items (e.g., people), with replacement.

    Args:
        max_n (int): the total number of items
        n (int): the number of items to choose

    **Example**::

        choices = cv.choose_r(5, 10) # choose 10 out of 5 people with equal probability (with repeats)
    '''
    return np.random.choice(max_n, n, replace=True)


def n_multinomial(probs, n): # No speed gain from Numba
    '''
    An array of multinomial trials.

    Args:
        probs (array): probability of each outcome, which usually should sum to 1
        n (int): number of trials

    Returns:
        Array of integer outcomes

    **Example**::

        outcomes = cv.multinomial(np.ones(6)/6.0, 50)+1 # Return 50 die-rolls
    '''
    return np.searchsorted(np.cumsum(probs), np.random.random(n))


@nb.njit((nbfloat,), cache=cache) # Numba hugely increases performance
def poisson(rate):
    '''
    A Poisson trial.

    Args:
        rate (float): the rate of the Poisson process

    **Example**::

        outcome = cv.poisson(100) # Single Poisson trial with mean 100
    '''
    return np.random.poisson(rate, 1)[0]


@nb.njit((nbfloat, nbint), cache=cache) # Numba hugely increases performance
def n_poisson(rate, n):
    '''
    An array of Poisson trials.

    Args:
        rate (float): the rate of the Poisson process (mean)
        n (int): number of trials

    **Example**::

        outcomes = cv.n_poisson(100, 20) # 20 Poisson trials with mean 100
    '''
    return np.random.poisson(rate, n)


def n_neg_binomial(rate, dispersion, n, step=1): # Numba not used due to incompatible implementation
    '''
    An array of negative binomial trials. See cv.sample() for more explanation.

    Args:
        rate (float): the rate of the process (mean, same as Poisson)
        dispersion (float):  dispersion parameter; lower is more dispersion, i.e. 0 = infinite, ∞ = Poisson
        n (int): number of trials
        step (float): the step size to use if non-integer outputs are desired

    **Example**::

        outcomes = cv.n_neg_binomial(100, 1, 50) # 50 negative binomial trials with mean 100 and dispersion roughly equal to mean (large-mean limit)
        outcomes = cv.n_neg_binomial(1, 100, 20) # 20 negative binomial trials with mean 1 and dispersion still roughly equal to mean (approximately Poisson)
    '''
    nbn_n = dispersion
    nbn_p = dispersion/(rate/step + dispersion)
    samples = np.random.negative_binomial(n=nbn_n, p=nbn_p, size=n)*step
    return samples