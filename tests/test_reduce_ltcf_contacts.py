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

    # Create a population with LTCF
    population = sp.make_population(n, generate=generate, with_facilities=with_facilities, use_two_group_reduction=True, average_LTCF_degree=20)

    # Check to see if all residents are in contact with at least one staff member
    sp.check_all_residents_are_connected_to_staff(population)
