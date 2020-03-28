import synthpops as sp
import sciris as sc
import numpy as np
import pandas as pd
import functools
import math
import os, sys
from copy import deepcopy


if __name__ == '__main__':

    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    popdict = {}

    n = 2500

    use_bayesian = True
    # use_bayesian = False

    options_args = {'use_microstructure': True, 'use_bayesian': use_bayesian}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict,state_location = state_location,location = location, options_args = options_args, network_distr_args = network_distr_args)

    uids = contacts.keys()
    uids = [uid for uid in uids]
    print(contacts[uids[22]]['contacts'])

    contacts = sp.trim_contacts(contacts,trimmed_size_dic=None,use_clusters=False)
    print(contacts[uids[22]]['contacts'])
