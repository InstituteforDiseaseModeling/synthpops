import sciris as sc
import synthpops as sp
import pandas as pd
import numpy as np
import os


def load_data_files(datadir,location,state_location,country_location,Npop,use_bayesian=True):
    if use_bayesian:
        file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location)
    else:
        file_path = os.path.join(datadir,'demographics',country_location,state_location)

    households_by_uid_path = os.path.join(file_path,location + '_' + str(Npop) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path,location + '_' + str(Npop) + '_age_by_uid.dat')

    workplaces_by_uid_path = os.path.join(file_path,location + '_' + str(Npop) + '_synthetic_workplaces_with_uids.dat')
    schools_by_uid_path = os.path.join(file_path,location + '_' + str(Npop) + '_synthetic_schools_with_uids.dat')

    df = pd.read_csv(age_by_uid_path, delimiter = ' ',header = None)

    age_by_uid_dic = dict(zip( df.iloc[:,0], df.iloc[:,1]))
    uids = age_by_uid_dic.keys()

    # you have both ages and sexes so we'll just populate that for you...
    popdict = {}
    for i,uid in enumerate(uids):
        popdict[uid] = {}
        popdict[uid]['age'] = int(age_by_uid_dic[uid])
        popdict[uid]['sex'] = np.random.binomial(1,p=0.5)
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {}
        for k in ['H','S','W','R']:
            popdict[uid]['contacts'][k] = set()

    households = []
    fh = open(households_by_uid_path,'r')
    for c,line in enumerate(fh):
        r = line.strip().split(' ')
        households.append(r)
        # for uid in r:

        #     for juid in r:
        #         if uid != juid:
        #             popdict[uid]['contacts']['H'].add(juid)
    fh.close()

    schools = []
    fs = open(schools_by_uid_path,'r')
    for c,line in enumerate(fs):
        r = line.strip().split(' ')
        schools.append(r)
        # for uid in r:
        #     for juid in r:
        #         if uid != juid:
        #             popdict[uid]['contacts']['S'].add(juid)
    fs.close()

    workplaces = []
    fw = open(workplaces_by_uid_path,'r')
    for c,line in enumerate(fw):
        r = line.strip().split(' ')
        workplaces.append(r)
        # for uid in r:
        #     for juid in r:
        #         if uid != juid:
        #             popdict[uid]['contacts']['W'].add(juid)
    fw.close()

    data = {}
    data['popdict'] = popdict
    data['households'] = households
    data['schools'] = schools
    data['workplaces'] = workplaces

    return data


def rehydrate(data):
    popdict = sc.dcp(data['popdict'])
    mapping = {'H':'households', 'S':'schools', 'W':'workplaces'}
    for key,label in mapping.items():
        for r in data[label]: # House, school etc
            for uid in r:
                for juid in r:
                    if uid != juid:
                        popdict[uid]['contacts'][key].add(juid)
    return popdict


dosave = True
savedir = '../data'

sc.tic()


state_location = 'Washington'
location = 'seattle_metro'
country_location = 'usa'

options_args = {'use_microstructure': True}
folder = os.path.join(sp.datadir, 'demographics', 'contact_matrices_152_countries', 'Washington')

datasets = sc.objdict()

for Npop in sp.api.popsize_choices:
    print(f'Loading {Npop}...')

    popdict1 = sp.make_contacts_from_microstructure(sp.datadir,location,state_location,country_location,Npop,use_bayesian=True)

    data = load_data_files(sp.datadir,location,state_location,country_location,Npop,use_bayesian=True)

    popdict2 = rehydrate(data)

    datasets[f'n{Npop}'] = data

    if dosave:
        sc.saveobj(f'{savedir}/synthpop_{Npop}.pop', data)

sc.toc()
print('Done.')