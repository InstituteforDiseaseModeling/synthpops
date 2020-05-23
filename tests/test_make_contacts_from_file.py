import synthpops as sp
import numpy as np


if __name__ == '__main__':
    
    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    n = 100e3

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}

    population = sp.make_contacts(location=location, state_location=state_location, country_location=country_location, options_args=options_args, network_distr_args=network_distr_args)
