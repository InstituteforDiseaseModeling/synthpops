"""
An example of creating a population with LTCF and reducing contacts within LTCF
while ensuring that every resident is in contact with at least one staff member.
"""

import numpy as np
import synthpops as sp

set_seed = True
set_seed = False

if set_seed:
    seed = 70
    np.random.seed(seed)

datadir = sp.datadir
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'


def test_create_reduced_contacts_with_group_types():
    n = 10000
    average_LTCF_degree = 20

    # First create contact_networks_facilities
    # set write to False and instead use return_popdict = True to get a population dict
    popdict = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location,
                                                         n, school_enrollment_counts_available=True,
                                                         write=False, do_plot=False, return_popdict=True)

    # Make 2 groups of contacts
    # Facility contacts

    contacts_group_1 = popdict  # with write = True the next line is creating a copy of popdict but from file.
    # contacts_group_1 = sp.make_contacts_with_facilities_from_microstructure(datadir, location, state_location,
                                                                            # country_location,
                                                                            # n)
    contacts_group_1_list = []
    uids = contacts_group_1.keys()
    uids = [uid for uid in uids]
    for n, uid in enumerate(uids):
        layers = contacts_group_1[uid]['contacts']['LTCF']  # this is a set in synthpops
        contacts_group_1_list.append(list(layers))  # maybe append a list form rather than set?
    print(*contacts_group_1_list)  # prints the list of lists of contacts in the LTCF
    print(len(contacts_group_1_list))

    contacts_group_1_list = uids[0:30]  # grab a subsample

    # Home contacts
    network_distr_args = {'average_degree': average_LTCF_degree, 'network_type': 'poisson_degree', 'directed': True}
    contacts_group_2 = sp.make_contacts_generic(popdict, network_distr_args=network_distr_args)
    contacts_group_2_list = []
    # for n, uid in contacts_group_2.items():  # this enumerates the dictionaries for all people in the second graph, results in a list of dictionaries
    for n, uid in enumerate(contacts_group_2.keys()):  #  want to the reduced contacts method below a list of uids
        contacts_group_2_list.append(uid)
    print(*contacts_group_2_list)
    print(len(contacts_group_2_list))

    # Now reduce contacts
    # won't work because contacts_group_1 and contacts_group_2 are dictionaries
    # reduced_contacts = sp.create_reduced_contacts_with_group_types(popdict, contacts_group_1, contacts_group_2, 'LTCF',
                                                                   # average_degree=average_LTCF_degree,
                                                                   # force_cross_edges=True)

    # probably won't work to use all uids possible in the first group because number of uids is too big or just take a long time
    # reduced_contacts = sp.create_reduced_contacts_with_group_types(popdict, uids, contacts_group_2_list, 'LTCF',
                                                                   # average_degree=average_LTCF_degree,
                                                                   # force_cross_edges=True)

    # works because contacts_group_1 and contacts_group_2 are lists
    reduced_contacts = sp.create_reduced_contacts_with_group_types(popdict, contacts_group_1_list, contacts_group_2_list, 'LTCF',
                                                                   average_degree=average_LTCF_degree,
                                                                   force_cross_edges=True)

    return reduced_contacts


if __name__ == '__main__':

    datadir = sp.datadir
    country_location = 'usa'
    state_location = 'Washington'
    location = 'seattle_metro'
    sheet_name = 'United States of America'

    with_facilities = True
    with_industry_code = False
    generate = True

    n = 2.5e3
    n = int(n)

    options_args = {'use_microstructure': True, 'use_industry_code': with_industry_code, 'use_long_term_care_facilities': with_facilities}
    network_distr_args = {'Npop': int(n)}

    k = 20

    # # Create a population with LTCF
    # population = sp.make_population(n, generate=generate, with_facilities=with_facilities, use_two_group_reduction=True, average_LTCF_degree=20)
    #
    # # Check to see if all residents are in contact with at least one staff member
    # sp.check_all_residents_are_connected_to_staff(population)

    create_reduced_contacts_with_group_types = test_create_reduced_contacts_with_group_types()
