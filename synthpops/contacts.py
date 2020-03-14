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

    # UIDs were supplied, use them
    if uids is not None:
        n = len(uids)

    if n < 3000:
        raise NotImplementedError("Stop! I can't work with fewer than 3000 people currently.")

    # Not supplied, generate
    if uids is None:
        uids = []
        for i in range(n):
            uids.append(sc.uuid(length=id_len))

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
    for i,uid in enumerate(uids):
        popdict[uid] = {}
        popdict[uid]['age'] = ages[i]
        popdict[uid]['sex'] = sexes[i]
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {'M': set()}

    return popdict


def make_contacts(popdict, weights_dic, n_contacts=30, use_age=True, use_sex=True,
                  use_loc=False, use_social_layers=True, use_student_weights=True,
                  student_age_min=3, student_age_max=20, worker_age_min=20,
                  worker_age_max=70, student_teacher_ratio=30, directed=False,
                  use_seattle=True):
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

    if not use_student_weights:
        raise NotImplementedError

    popdict = sc.dcp(popdict) # To avoid modifying in-place


    if use_seattle and not use_social_layers:
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

                if directed:
                    for i in popdict:

                        nc = sp.pt(n_contacts)
                        contact_ages = sp.sample_n_contact_ages(nc,popdict[i]['age'],age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic,num_agebrackets)
                        popdict[i]['contacts']['M'] = popdict[i]['contacts']['M'].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

                elif not directed:
                    for i in popdict:
                        nc = sp.pt(n_contacts/2)
                        contact_ages = sp.sample_n_contact_ages(nc,popdict[i]['age'],age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic,num_agebrackets)
                        popdict[i]['contacts']['M'] = popdict[i]['contacts']['M'].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

                        for c in popdict[i]['contacts']['M']:
                            popdict[c]['contacts']['M'].add(i)


    if use_seattle and use_social_layers:
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

                # weights_dic is calibrated to empirical survey data but for all people by all ages!
                # to figure out the weights for individual ranges, this is an approach to rescale school and workplace weights
                # really a guess of what it should be - will fix to be flexible later

                student_weights_dic = sc.dcp(weights_dic)
                non_student_weights_dic = sc.dcp(weights_dic)
                if use_student_weights:

                    n_students = np.sum([len(uids_by_age_dic[a]) for a in range(student_age_min,student_age_max+1)])
                    n_workers = np.sum([len(uids_by_age_dic[a]) for a in range(worker_age_min,worker_age_max+1)])
                    n_teachers = n_students/student_teacher_ratio
                    teachers_school_weight = n_teachers/n_workers

                    student_weights_dic['S'] = student_weights_dic['S'] + student_weights_dic['W']
                    student_weights_dic['W'] = 0

                    non_student_weights_dic['S'] = teachers_school_weight
                    non_student_weights_dic['W'] = non_student_weights_dic['W'] + non_student_weights_dic['S'] - teachers_school_weight

                    student_weights_dic = sp.norm_dic(student_weights_dic)
                    non_student_weights_dic = sp.norm_dic(non_student_weights_dic)


                if directed:

                    for uid in popdict:
                        for k in weights_dic:
                            popdict[uid]['contacts'][k] = set()

                    for uid in popdict:

                        for k in weights_dic:
                            if popdict[uid]['age'] <= 20:
                                nc = sp.pt(n_contacts * student_weights_dic[k])
                            elif popdict[uid]['age'] > 20:
                                nc = sp.pt(n_contacts * non_student_weights_dic[k])

                            contact_ages = sp.sample_n_contact_ages(nc,popdict[uid]['age'],age_brackets,age_by_brackets_dic, age_mixing_matrix_dic, {k:1})
                            popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

                elif not directed:
                    for uid in popdict:
                        for k in weights_dic:
                            popdict[uid]['contacts'][k] = set()

                    for uid in popdict:

                        for k in weights_dic:
                            if popdict[uid]['age'] <= 20:
                                nc = sp.pt(n_contacts / 2 * student_weights_dic[k])
                            elif popdict[uid]['age'] > 20:
                                nc = sp.pt(n_contacts / 2 * non_student_weights_dic[k])

                            contact_ages = sp.sample_n_contact_ages(nc,popdict[uid]['age'],age_brackets,age_by_brackets_dic, age_mixing_matrix_dic, {k: 1})
                            popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                            for c in popdict[uid]['contacts'][k]:
                                popdict[c]['contacts'][k].add(uid)


    return popdict


