import os
import numpy as np
import synthpops as sp
import sciris as sc


if __name__ == '__main__':

    sp.validate()
    datadir = sp.datadir
    
    location = 'seattle_metro'
    state_location = 'Washington'

    # location = 'portland_metro'
    # state_location = 'Oregon'
    country_location = 'usa'

    # sp.write_16_age_brackets_distr(location,state_location,country_location)