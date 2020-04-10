import sciris as sc
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
import os
import synthpops as sp

from copy import deepcopy
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean


def generate_synthetic_population_with_workplace_industries(n,datadir,location='seattle_metro',state_location='Washington',country_location='usa',sheet_name='United States of America',level='county',verbose=False,plot=False):

    age_brackets = sp.get_census_age_brackets(datadir,state_location,country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    num_agebrackets = len(age_brackets)
    contact_matrix_dic = sp.get_contact_matrix_dic(datadir,sheet_name)

    household_size_distr = sp.get_household_size_distr(datadir,location,state_location,country_location)

    if n < 5000:
        raise NotImplementedError("Population is too small to currently be generated properly. Try a size larger than 5000.")

    # this could be unnecessary if we get the single year age distribution in a different way.
    n_to_sample_smoothly = int(1e6)
    # hh_sizes = sp.generate_household_sizes(n_to_sample_smoothly,household_size_distr)
    # hh_sizes = sp.generate_household_sizes(n_to_sample_smoothly,household_size_distr)
    # sp.trim_households(1000,household_size_distr)
    sp.make_popdict(n=5000)
