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

    n = 20000

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}

    sc.tic()
    contacts = sp.make_contacts(popdict,state_location = state_location,location = location, options_args = options_args, network_distr_args = network_distr_args)
    # uids = contacts.keys()
    # uids = [uid for uid in uids]
    # print(contacts[uids[3]]['contacts'])

    # contacts = sp.trim_contacts(contacts,trimmed_size_dic=None,use_clusters=False)
    # print(contacts[uids[3]]['contacts'])



    sp.save_synthpop(os.path.join(datadir,'demographics',country_location,state_location),contacts)

    sc.toc()

