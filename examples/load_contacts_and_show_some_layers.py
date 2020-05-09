"""
An example of how to load synthetic populations with microstructure (households, schools, and workplaces)
Populations have demographics (age, sex) from data.
Not an exhaustive list of what synthpops can do - please take a look through the code base for the many possibilities.
"""

import synthpops as sp
import sciris as sc
import numpy as np
import os

def show_layers(popdict,show_ages=False):

    uids = popdict.keys()
    uids = [uid for uid in uids]

    layers = popdict[uids[0]]['contacts'].keys()
    if show_ages:
        for uid in uids:
            print(uid,popdict[uid]['age'])
            for k in layers:
                contact_ages = [popdict[c]['age'] for c in popdict[uid]['contacts'][k]]
                print(k,sorted(contact_ages))

    else:
        for uid in uids:
            print(uid)
            for k in layers:
                print(k,contacts[uid]['contacts'][k])


if __name__ == '__main__':

    datadir = sp.datadir # point datadir where your data folder lives

    # location information - currently we only support the Seattle Metro area in full, however other locations can be supported with this framework at a later date
    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'
    level = 'county'

    n = 20000
    verbose = True
    plot = True

    # loads population with microstructure and age demographics that approximate those of the location selected
    # files located in:
    #    datadir/demographics/contact_matrices_152_countries/state_location/

    # load population into a dictionary of individuals who know who their contacts are
    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': n}
    contacts = sp.make_contacts(location=location,state_location=state_location,country_location=country_location,options_args=options_args,network_distr_args=network_distr_args)

    # not all school and workplace contacts are going to be close contacts so create 'closer' contacts for these settings
    close_contacts_number = {'S': 20, 'W': 20}
    contacts = sp.trim_contacts(contacts,trimmed_size_dic=close_contacts_number)


    verbose = True
    # verbose = False
    if verbose:
        show_layers(contacts,show_ages=True)


