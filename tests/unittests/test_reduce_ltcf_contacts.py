print('Deprecated, see test_long_term_care_facilities.py instead')

# """
# An example of creating a population with LTCF and reducing contacts within LTCF
# while ensuring that every resident is in contact with at least one staff member.
# """

# import synthpops as sp

# set_seed = True

# if set_seed:
#     seed = 70
#     sp.set_seed(seed)

# datadir = sp.datadir
# location = 'seattle_metro'
# state_location = 'Washington'
# country_location = 'usa'


# def test_create_reduced_contacts_with_group_types():
#     n = 1001
#     average_LTCF_degree = 20

#     # First create contact_networks_facilities
#     # set write to False and instead use return_popdict = True to get a population dict
#     popdict = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location,
#                                                          n,
#                                                          write=False, plot=False, return_popdict=True)

#     # Make 2 groups of contacts
#     # Facility contacts - use generating function so that data can be generated as needed to run this test
#     contacts_group_1 = sp.generate_microstructure_with_facilities(datadir, location, state_location, country_location,
#                                                                   n, plot=False,
#                                                                   write=False, return_popdict=True,
#                                                                   use_two_group_reduction=False, average_LTCF_degree=20)

#     # List of ids for group_1 facility contacts
#     contacts_group_1_list = list(contacts_group_1.keys())

#     # Home contacts
#     network_distr_args = {'average_degree': average_LTCF_degree, 'network_type': 'poisson_degree', 'directed': True}
#     contacts_group_2 = sp.make_contacts_generic(popdict, network_distr_args=network_distr_args)
#     # List of ids for group_2 home contacts
#     contacts_group_2_list = list(contacts_group_2.keys())

#     # Now reduce contacts
#     reduced_contacts = sp.create_reduced_contacts_with_group_types(popdict, contacts_group_1_list,
#                                                                    contacts_group_2_list, 'LTCF',
#                                                                    average_degree=average_LTCF_degree,
#                                                                    force_cross_edges=True)

#     assert len(reduced_contacts)*2 == len(contacts_group_1_list) + len(contacts_group_2_list)

#     for i in popdict:
#         person = reduced_contacts[i]
#         if person['snf_res'] == 1:

#             contacts = person['contacts']['LTCF']
#             staff_contacts = [j for j in contacts if popdict[j]['snf_staff'] == 1]

#             if len(staff_contacts) == 0:
#                 errormsg = f'At least one LTCF or Skilled Nursing Facility resident has no contacts with staff members.'
#                 raise ValueError(errormsg)

#     return reduced_contacts


# if __name__ == '__main__':
#     # datadir = sp.datadir
#     country_location = 'usa'
#     state_location = 'Washington'
#     location = 'seattle_metro'
#     sheet_name = 'United States of America'

#     with_facilities = True
#     with_industry_code = False
#     generate = True

#     n = 1000
#     n = int(n)

#     options_args = {'use_microstructure': True, 'use_industry_code': with_industry_code,
#                     'use_long_term_care_facilities': with_facilities}
#     network_distr_args = {'Npop': int(n)}

#     k = 20

#     # # Create a population with LTCF
#     population = sp.make_population(n, generate=generate, with_facilities=with_facilities, use_two_group_reduction=True,
#                                     average_LTCF_degree=20)

#     # Check to see if all residents are in contact with at least one staff member
#     sp.check_all_residents_are_connected_to_staff(population)

#     # Create reduced contacts from 2 groups of contacts
#     create_reduced_contacts_with_group_types = test_create_reduced_contacts_with_group_types()
