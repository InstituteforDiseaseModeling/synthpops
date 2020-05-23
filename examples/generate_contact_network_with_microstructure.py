import synthpops as sp
import sciris as sc
import numpy as np
import os

"""
An of how to make synthetic populations with microstructure (households, schools, and workplaces)
Populations have demographics (age, sex) from data.
Not an exhaustive list of what synthpops can do - please take a look through the code base for the many possibilities.
"""

# def show_layers(popdict,show_ages=False):

#     uids = popdict.keys()
#     uids = [uid for uid in uids]

#     layers = popdict[uids[0]]['contacts'].keys()
#     if show_ages:
#         for uid in uids:
#             print(uid,popdict[uid]['age'])
#             for k in layers:
#                 contact_ages = [popdict[c]['age'] for c in popdict[uid]['contacts'][k]]
#                 print(k,sorted(contact_ages))

#     else:
#         for uid in uids:
#             print(uid)
#             for k in layers:
#                 print(k,contacts[uid]['contacts'][k])


if __name__ == '__main__':

    datadir = sp.datadir # point datadir where your data folder lives

    # location information - currently we only support the Seattle Metro area in full, however other locations can be supported with this framework at a later date
    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 11000
    verbose = False
    plot = True
    write = True

    # this will generate a population with microstructure and age demographics that approximate those of the location selected
    # also saves to file in:
    #    datadir/demographics/contact_matrices_152_countries/state_location/
    sp.generate_synthetic_population(n,datadir,location=location,state_location=state_location,country_location=country_location,sheet_name=sheet_name,verbose=verbose,plot=plot,write=write)

    # load that population into a dictionary of individuals who know who their contacts are
    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': n}
    contacts = sp.make_contacts(location=location,state_location=state_location,country_location=country_location,options_args=options_args,network_distr_args=network_distr_args)

    verbose = True
    # verbose = False
    if verbose:
        sp.show_layers(contacts,show_ages=True, show_n=2)
