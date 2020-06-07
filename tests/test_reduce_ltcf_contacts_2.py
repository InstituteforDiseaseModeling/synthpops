"""
An example of creating a population with LTCF and reducing contacts within LTCF
while ensuring that every resident is in contact with at least one staff member. 
"""

import numpy as np
import synthpops as sp
import sciris as sc

set_seed = True
# set_seed = False

if set_seed:
    seed = 70
    np.random.seed(seed)

datadir = sp.datadir
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'


def check_reduced_contacts_with_group_types(popdict, layer, group_1, group_2):
    group = list(group_1) + list(group_2)

    for i in group:
        if i in group_1:
            layer_contacts = popdict[i]['contacts'][layer]
            layer_contacts_in_2 = layer_contacts.intersection(set(group_2))

            if len(layer_contacts_in_2) == 0:
                errormsg = f'At least one person in group 1 has no contacts with group 2.'
                raise ValueError(errormsg)
            # print(i, layer_contacts_in_2)

    print('Everyone in group 1 has at least one contact in group 2')


def test_create_reduced_contacts_with_group_types():
    n = 200
    average_LTCF_degree = 3

    # First create contact_networks_facilities
    popdict = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location,
                                                         n, school_enrollment_counts_available=False,
                                                         write=False, plot=False, return_popdict=True)

    # Make 2 groups of contacts
    uids = popdict.keys()
    uids = [uid for uid in uids]

    # # keys for each person from microstructure with facilities populations
    # attributes = popdict[uids[0]].keys()

    # One group of those who live in any of the facilities
    ltcf_residents = []

    for u, uid in enumerate(uids):
        person = popdict[uid]
        if person['snf_res']:
            ltcf_residents.append(uid)

    # Second group from another popdict
    network_distr_args = {'average_degree': average_LTCF_degree, 'network_type': 'poisson_degree', 'directed': True}

    # size
    n2 = 50

    # set up the ids so the second population doesn't have uids that overlap with the first population
    uids_2 = [i for i in range(n, n+n2)]

    # initialize the second population to have ages and uids
    popdict_2 = sp.make_popdict(n=n2, uids=uids_2, country_location=country_location, use_demography=False)

    # create a network for the second population with defined network properties
    popdict_2 = sp.make_contacts_generic(popdict_2, network_distr_args=network_distr_args)

    popdict = sc.mergedicts(popdict_2, popdict)
    # print(popdict[uids[n2-1]])
    print(popdict_2)
    new_layer = 'new_layer'
    reduced_contacts = sp.create_reduced_contacts_with_group_types(popdict, ltcf_residents, uids_2, new_layer,
                                                                   average_degree=average_LTCF_degree,
                                                                   force_cross_edges=True)

    check_reduced_contacts_with_group_types(popdict, new_layer, ltcf_residents, uids_2)


if __name__ == '__main__':

    test_create_reduced_contacts_with_group_types()
