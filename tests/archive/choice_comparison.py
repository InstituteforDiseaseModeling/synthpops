import numpy   as np; np
import pylab   as pl; pl
import pandas  as pd; pd
import sciris  as sc; sc
import covasim as cv; cv
import random
import random as rnd

import itertools
import bisect as bs
import bisect


def fast_choices(weights):
    """
    Choose an option -- quickly -- from the provided weights.

    Reimplementation of random.choices() removing the junk.
    """
    cum_weights = list(itertools.accumulate(weights))
    total = cum_weights[-1]
    max_ind = len(cum_weights) - 1
    return bs.bisect(cum_weights, random._inst.random()*total, 0, max_ind)


def fast_choice(weights):
    """
    Choose an option -- quickly -- from the provided weights. Weights do not need
    to be normalized.

    Reimplementation of random.choices(), removing the junk.

    Example:
        fast_choice([0.1,0.2,0.3,0.2,0.1]) # might return 2
    """
    cum_weights = list(itertools.accumulate(weights))
    if cum_weights[-1] == 0:
        return 0
    return bisect.bisect(cum_weights, random._inst.random()*(cum_weights[-1]), 0, len(cum_weights)-1)


repeats = int(1e4)

norm_distr = [1, 2, 3, 2, 1]
norm_sorted_distr = np.array(norm_distr)
norm_sorted_distr = norm_sorted_distr/norm_sorted_distr.sum()
n_choices = len(norm_distr)
choices = list(range(len(norm_distr)))
cum_distr = np.cumsum(norm_distr).tolist()


res1 = np.zeros(repeats)
res2 = np.zeros(repeats)
res3 = np.zeros(repeats)
res4 = np.zeros(repeats)
res5 = np.zeros(repeats)
res6 = np.zeros(repeats)


N = 5

print('\nmultinomial')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            n = np.random.multinomial(1, norm_sorted_distr, size=1)[0]
            res1[r] = np.where(n)[0][0]

print('\nnp.random.choice')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            res2[r] = np.random.choice(n_choices, 1, p=norm_sorted_distr)

print('\nrnd.choices, cheating')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            res3[r] = rnd.choices(choices, cum_weights=cum_distr)[0]

print('\nrnd.choices, real')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            res4[r] = rnd.choices(list(range(len(norm_distr))), weights=norm_distr)[0]

print('\nfast_choices')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            res5[r] = fast_choices(norm_distr)

print('\nfast_choice')
for Q in range(N):
    with sc.Timer():
        for r in range(repeats):
            res6[r] = fast_choice(norm_distr)


pl.subplot(3,2,1)
pl.hist(res1)

pl.subplot(3,2,2)
pl.hist(res2)

pl.subplot(3,2,3)
pl.hist(res3)

pl.subplot(3,2,4)
pl.hist(res4)

pl.subplot(3,2,5)
pl.hist(res5)

pl.subplot(3,2,6)
pl.hist(res6)