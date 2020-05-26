import synthpops as sp
import sciris as sc
import numpy as np
import os

import pandas as pd

"""
Save synthpop_n.pop objects for different population sizes created n.
"""

if __name__ == '__main__':
    
    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    n = 5000

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}

    contacts = sp.make_contacts(location=location,state_location=state_location,country_location=country_location,options_args=options_args,network_distr_args=network_distr_args)

    # save to file
    # file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'contact_networks')
    # sp.save_synthpop(file_path,contacts,location)

    pop = contacts
    keys = pop.keys()
    keys = [k for k in keys]
    sc.tic()
    pop = sp.make_popdict(n, uids=keys, location=location, state_location=state_location, country_location=country_location, use_demography=True)
    sc.toc()
    # print(pop)s
    # print(keys)

    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')

    # households_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_age_by_uid.dat')
    df = pd.read_csv(age_by_uid_path, delimiter=' ', header=None)


    # age_by_uid_dic = sc.objdict(zip(df.iloc[:,0], df.iloc[:,1]))
    age_by_uid_dic = sc.objdict(dict(zip(df.iloc[:, 0], df.iloc[:, 1].astype(int))))
    # age_by_uid_dic = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    uids = age_by_uid_dic.keys()

    # print(uids[0])
    print(age_by_uid_dic)
    print(uids[0], age_by_uid_dic[uids[0]], age_by_uid_dic[0])

    # sc.tic()
    # popdict = sp.make_contacts_from_microstructure(datadir, location, state_location, country_location, n)
    # sc.toc()
    # sp.show_layers(popdict)