import synthpops as sp
import sciris as sc
import numpy as np
import math
import os, sys
from copy import deepcopy


if __name__ == '__main__':
    
    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'
    level = 'county'

    n = 6000
    verbose = True
    plot = True

    hh_size_distr = {1:0.5, 2:0.5}
    sp.generate_synthetic_population_with_workplace_industries(n,datadir,location,state_location,country_location,sheet_name,level,verbose,plot)
    # sp.generate_household_sizes(1000,{1:0.5,2:0.5})
    # sp.make_popdict(n)

    # sp.generate_synthetic_population(n,datadir,location=location,state_location=state_location,country_location=country_location,sheet_name=sheet_name,level=level,verbose=verbose,plot=plot)
    # sp.get_employment_rates(datadir,location,state_location,country_location)
    # sp.generate_household_sizes(n,hh_size_distr)