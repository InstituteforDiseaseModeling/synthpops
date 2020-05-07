import synthpops as sp
import sciris as sc
import numpy as np
import os

"""
Save synthpop_n.pop objects for different population sizes created n.
"""

if __name__ == '__main__':
    
    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    n = 120000

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}

    contacts = sp.make_contacts(location=location,state_location=state_location,country_location=country_location,options_args=options_args,network_distr_args=network_distr_args)

    # save to file
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'contact_networks')
    sp.save_synthpop(file_path,contacts,location)
