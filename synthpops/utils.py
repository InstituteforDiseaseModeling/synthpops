import os
import platform
import pandas as pd
import numba as nb
import numpy as np
import random


__all__ = ['set_seed']


def get_kc_snf_df_location():

    os_platform = platform.system()
    userpath = os.path.expanduser('~')
    username = os.path.split(userpath)[-1]
    if os_platform == "Windows":
        KC_snf_df = pd.read_csv(
            os.path.join('C:/Users', username, 'Dropbox (IDM)', 'COVID-19', 'seattle_network', 'secure_King_County',
                         'IDM_CASE_FACILITY.csv'))
    elif os_platform != "Windows":
        KC_snf_df = pd.read_csv(
            os.path.join('/home', username, 'Dropbox (IDM)', 'COVID-19', 'seattle_network', 'secure_King_County',
                         'IDM_CASE_FACILITY.csv'))

    return KC_snf_df


def set_seed(seed=None):
    ''' Reset the random seed -- complicated because of Numba '''

    @nb.njit((nb.int64,), cache=True)
    def set_seed_numba(seed):
        return np.random.seed(seed)

    def set_seed_regular(seed):
        return np.random.seed(seed)

    # Dies if a float is given
    if seed is not None:
        seed = int(seed)

    set_seed_regular(seed) # If None, reinitializes it
    if seed is None: # Numba can't accept a None seed, so use our just-reinitialized Numpy stream to generate one
        seed = np.random.randint(1e9)
    set_seed_numba(seed)
    random.seed(seed) # Finally, reset Python's built-in random number generator

    return