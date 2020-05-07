import synthpops as sp
import sciris as sc
import numpy as np
import math
import os, sys
from copy import deepcopy


if __name__ == '__main__':
    sp.validate()
    # datadir = sp.datadir
    datadir = sp.set_datadir(sp.config.localdatadir)  # set to local data directory

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 12000
    verbose = True
    verbose = False
    # plot = True
    plot = False
    school_enrollment_counts_available = False
    use_default = False

    sp.generate_synthetic_population(n, datadir, location=location, state_location=state_location, country_location=country_location, sheet_name=sheet_name, school_enrollment_counts_available=school_enrollment_counts_available, verbose=verbose, plot=plot, use_default=use_default)
