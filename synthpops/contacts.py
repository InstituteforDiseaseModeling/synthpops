import sciris as sc
import numpy as np
from . import synthpops as sp
from .config import datadir
import os


def make_popdict(n=None, uids=None, ages=None, sexes=None, use_seattle=True, id_len=6):
    """ Create a dictionary of n people with age, sex and loc keys """
    
    
    # A list of UIDs was supplied as the first argument
    if isinstance(n, list):
        uids = n
        n = len(uids)
    elif uids is not None: # UIDs were supplied, use them
        n = len(uids)
    
    # Not supplied, generate
    if uids is None:
        uids = []
        for i in range(n):
            uids.append(str(sc.uuid())[:id_len])
    
    # Optionally take in either aes or sexes, too
    if ages is None or sexes is None:
        if use_seattle:
            gen_ages,gen_sexes = sp.get_seattle_age_sex_n(n_people = n)
        else:
            raise NotImplementedError('Currently, only Seattle is supported')
        random_inds = np.random.permutation(n)
        if ages is None:
            ages = [gen_ages[r] for r in random_inds]
        if sexes is None:
            sexes = [gen_sexes[r] for r in random_inds]

    popdict = {}
    for uid in uids:
        popdict[uid] = {}
        popdict[uid]['age'] = ages[i]
        popdict[uid]['sex'] = sexes[i]
        popdict[uid]['loc'] = None
        
    return popdict



def make_contacts(popdict,weights_dic,n_contacts=30, use_age=True, use_sex=True, use_loc=False, directed=False, use_seattle = True):
    '''
    Generates a list of contacts for everyone in the population. popdict is a
    dictionary with N keys (one for each person), with subkeys for age, sex, location,
    and potentially other factors. This function adds a new subkey, contacts, which
    is a list of contact IDs for each individual. If directed=False (default),
    if person A is a contact of person B, then person B is also a contact of person
    A.
    
    Example output (input is the same, minus the "contacts" field):
        popdict = {
            '8acf08f0': {
                'age': 57.3,
                'sex': 0,
                'loc': (47.6062, 122.3321),
                'contacts': ['43da76b5']
                },
            '43da76b5': {
                'age': 55.3,
                'sex': 1,
                'loc': (47.2473, 122.6482),
                'contacts': ['8acf08f0', '2d2ad46f']
                },
            '2d2ad46f': {
                'age': 27.3,
                'sex': 1,
                'loc': (47.2508, 122.1492),
                'contacts': ['43da76b5', '5ebc3740']
                },
            '5ebc3740': {
                'age': 28.8,
                'sex': 1,
                'loc': (47.6841, 122.2085),
                'contacts': ['2d2ad46f']
                },
        }
    '''
    
    popdict = sc.dcp(popdict) # To avoid modifyig in-place
    if use_seattle:
        if use_age:
            if not use_loc:

                uids_by_age_dic = sp.get_uids_by_age_dic(popdict)

                dropbox_path = datadir
                # census_location = 'seattle_metro'
                location = 'Washington'
                num_agebrackets = 18

                # age_bracket_distr = sp.read_age_bracket_distr(dropbox_path, census_location)

                # gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path, census_location)

                age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
                age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
                age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

                age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path,location,num_agebrackets)
                age_mixing_matrix_dic['M'] = sp.get_contact_matrix(dropbox_path,location,'M',num_agebrackets)

                for i in popdict:
                    nc = sp.pt(n_contacts)
                    # print(i,nc)
                    contact_ages = sp.sample_n_contact_ages(nc,popdict[i]['age'],age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic,num_agebrackets)
                    popdict[i]['contacts'] = sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic)
    
    
    return popdict


