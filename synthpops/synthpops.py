import os
import numpy as np
import pandas as pd
import sciris as sc
import numba  as nb
from collections import Counter
from copy import deepcopy
from .config import datadir


def norm_dic(dic):
    """
    Return normalized dict.
    """
    total = np.sum([dic[i] for i in dic], dtype = float)
    if total == 0.0:
        return dic
    new_dic = {}
    for i in dic:
        new_dic[i] = float(dic[i])/total
    return new_dic


def norm_age_group(age_dic,age_min,age_max):
    dic = {}
    if age_max == 100:
        age_max = 99
    for a in range(age_min,age_max+1):
        dic[a] = age_dic[a]
    return norm_dic(dic)


def get_age_brackets_from_df(ab_file_path):
    """
    Returns dict of age bracket ranges from ab_file_path.
    """
    ab_df = pd.read_csv(ab_file_path,header = None)
    dic = {}
    for index,row in enumerate(ab_df.iterrows()):
        age_min = row[1].values[0]
        age_max = row[1].values[1]
        dic[index] = np.arange(age_min,age_max+1)
    return dic


def get_gender_fraction_by_age_path(datadir, location=None, state_location=None):
    """
    Return file_path for gender fractions by age bracket. This should only be used if the data is available.
    """
    levels = [location,state_location]
    if any(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location, state_location, and country_location strings.")
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions',location + '_gender_fraction_by_age_bracket.dat')
        # return os.path.join(datadir,'demographics',country_location,state_location,'census','age distributions',location + '_gender_fraction_by_age_bracket.dat')


def read_gender_fraction_by_age_bracket(datadir, location=None, state_location=None, file_path=None):
    """
    Return dict of gender fractions by age bracket, either by location, state_location, country_location strings, or by the file_path if that's given.
    """
    if file_path is None:
        f = get_gender_fraction_by_age_path(datadir,location,state_location,file_path)
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path)
    dic = {}
    dic['male'] = dict(zip(np.arange(len(df)),df.fraction_male))
    dic['female'] = dict(zip(np.arange(len(df)),df.fraction_female))
    return dic


def get_age_bracket_distr_path(datadir, location=None, state_location=None):
    """
    Return file_path for age distribution by age brackets.
    """
    levels = [location,state_location]
    if any(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions',location + '_age_bracket_distr.dat')


def read_age_bracket_distr(datadir, location=None, state_location=None, file_path=None):
    """
    Return dict of age distribution by age brackets.
    """
    if file_path is None:
        file_path = get_age_bracket_distr_path(datadir,location,state_location)
    df = pd.read_csv(file_path)
    return dict(zip(np.arange(len(df)), df.percent))


def get_household_size_distr_path(datadir, location=None, state_location=None):
    """
    Return file_path for household size distribution
    """
    levels = [location,state_location]
    if any(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'household size distributions',location + '_household_size_distr.dat')


def get_household_size_distr(datadir, location=None, state_location=None, file_path=None):
    """
    Return a dictionary of the distributions of household sizes. If you don't give the file_path, then supply the location and state_location strings.
    """
    if file_path is None:
        file_path = get_household_size_distr_path(datadir,location,state_location)
    df = pd.read_csv(file_path)
    return dict( zip(df.household_size, df.percent) )


def get_head_age_brackets_path(datadir, state_location=None, country_location=None):
    """
    Return file_path for head of household age brackets. If data doesn't exist at the state level, only give the country_location. 
    """
    levels = [state_location,country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location is None:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,'household living arrangements','head_age_brackets.dat')
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'household living arrangements','head_age_brackets.dat')


def get_head_age_brackets(datadir, state_location=None,country_location=None, file_path=None):
    """
    Return head age brackets either from the file_path directly, or using the other parameters to figure out what the file_path should be.
    """
    if file_path is None:
        file_path = get_head_age_brackets_path(datadir,state_location,country_location)
    return get_age_brackets_from_df(file_path)


def get_household_head_age_by_size_path(datadir,state_location=None,country_location=None):
    """
    Return file_path for head of household age by size counts or distribution. If the data doesn't exist at the state level, only give the country_location.
    """
    levels = [state_location,country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location in None:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,'household living arrangements','household_head_age_and_size_count.dat')
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'household living arrangements','household_head_age_and_size_count.dat')


def get_household_head_age_by_size_df(datadir,state_location=None,country_location=None,file_path=None):
    """
    Return a pandas df of head of household age by the size of the household. If the file_path is given return from there first.
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir,state_location,country_location)
    return pd.read_csv(file_path)


def get_head_age_by_size_distr(datadir,state_location=None,country_location=None,file_path=None,household_size_1_included=False):
    """
    Return an array of head of household age bracket counts (col) given by size (row).
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir,state_location,country_location,file_path)
    hha_df = get_household_head_age_by_size_df(datadir,state_location,country_location,file_path)
    hha_by_size = np.zeros((2 + len(hha_df), len(hha_df.columns)-1))
    if household_size_1_included:
        for s in range(1, len(hha_df)+1):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    else:
        hha_by_size[0,:] += 1
        for s in range(2, len(hha_df)+2):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    return hha_by_size


def get_census_age_brackets_path(datadir,state_location=None,country_location=None):
    """
    Returns file_path for census age brackets: depends on the state or country of the source data on contact patterns.
    """
    levels = [state_location,country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location is None:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,'census_age_brackets.dat')
    else:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'census_age_brackets.dat')


def get_census_age_brackets(datadir,state_location=None,country_location=None,file_path=None):
    """
    Returns census age brackets: depends on the country or source of contact pattern data.
    """
    if file_path is None:
        file_path = get_census_age_brackets_path(datadir,state_location,country_location)

    return get_age_brackets_from_df(file_path)


def get_age_by_brackets_dic(age_brackets):
    """
    Returns dict of age bracket by age.
    """
    age_by_brackets_dic = {}
    for b in age_brackets:
        for a in age_brackets[b]:
            age_by_brackets_dic[a] = b
    return age_by_brackets_dic


def get_aggregate_ages(ages,age_by_brackets_dic):
    """
    Return an aggregate age count for specified age brackets (values in age_by_brackets_dic)
    """
    bracket_keys = set(age_by_brackets_dic.values())
    aggregate_ages = dict.fromkeys(bracket_keys,0)
    for a in ages:
        b = age_by_brackets_dic[a]
        aggregate_ages[b] += ages[a]
    return aggregate_ages


def get_aggregate_matrix(M,age_by_brackets_dic):
    """
    Return symmetric contact matrix aggregated to age brackets. Do not use for community (homogeneous) mixing matrix. 
    """
    N = len(M)
    M_agg = np.zeros((num_agebrackets,num_agebrackets))
    for i in range(N):
        bi = age_by_brackets_dic[i]
        for j in range(N):
            bj = age_by_brackets_dic[j]
            M_agg[bi][bj] += M[i][j]
    return M_agg


def get_asymmetric_matrix(symmetric_matrix,aggregate_ages):
    """
    Return asymmetric contact matrix from symmetric contact matrix. Now the element M_ij represents the number of contacts of age group j for the average individual of age group i.
    """
    M = deepcopy(symmetric_matrix)
    for a in aggregate_ages:
        M[a,:] = M[a,:]/float(aggregate_ages[a])

    return M


def get_aggregate_age_dict_conversion(larger_aggregate_ages,larger_age_brackets,smaller_age_brackets,age_by_brackets_dic_larger,age_by_brackets_dic_smaller):
    """
    Convert the aggregate age count in larger_aggregate_ages from a larger number of age brackets to a smaller number of age brackets
    """
    if len(larger_age_brackets) < len(smaller_age_brackets) : raise NotImplementedError('Cannot reduce aggregate ages any further.')
    smaller_aggregate_ages = dict.fromkeys(smaller_age_brackets.keys(),0)
    for lb in larger_age_brackets:
        a = larger_age_brackets[lb][0]
        sb = age_by_brackets_dic_smaller[a]
        smaller_aggregate_ages[sb] += larger_aggregate_ages[lb]
    return smaller_aggregate_ages


def write_16_age_brackets_distr(location,state_location,country_location):
    """
    Write to file age distribution to match the age brackets for contact matrices currently used by the webapp (from K. Prem et al).
    """
    census_age_bracket_distr = read_age_bracket_distr(datadir,location,state_location)
    census_age_brackets = get_census_age_brackets(datadir,country_location)
    webapp_age_brackets = get_census_age_brackets(datadir,'bayesian')

    census_age_by_brackets_dic = get_age_by_brackets_dic(census_age_brackets)
    webapp_age_by_brackets_dic = get_age_by_brackets_dic(webapp_age_brackets)

    webapp_age_bracket_distr = get_aggregate_age_dict_conversion(census_age_bracket_distr,census_age_brackets,webapp_age_brackets,census_age_by_brackets_dic,webapp_age_by_brackets_dic)

    fp = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions')
    os.makedirs(fp,exist_ok=True)
    f = open(os.path.join(fp,location + '_age_bracket_distr.dat'),'w')
    f.write('age_bracket,percent\n')
    for b in sorted(webapp_age_bracket_distr.keys()):
        sa,ea = webapp_age_brackets[b][0],webapp_age_brackets[b][-1]
        f.write(str(sa) + '_' + str(ea) + ',' + str(webapp_age_bracket_distr[b]) + '\n')
    f.close()


def get_symmetric_community_matrix(ages):
    """
    Return symmetric homogeneous community matrix for age count in ages.
    """
    N = len(ages)
    M = np.ones((N,N))
    for a in range(N):
        M[a,:] = M[a,:] * ages[a]
        M[:,a] = M[:,a] * ages[a]
    for a in range(N):
        M[a,a] -= ages[a]
    M = M/(np.sum([ages[a] for a in ages], dtype = float) - 1)
    return M


def get_contact_matrix(datadir,setting_code,sheet_name=None,file_path = None, delimiter = ' ', header = None):
    """
    Return setting specific contact matrix givn sheet name to use. If file_path is given, then delimiter and header should also be specified.
    """
    if file_path is None:
        setting_names = {'H': 'home', 'S': 'school', 'W': 'work', 'C': 'other_locations'}
        if setting_code in setting_names:
            file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries','MUestimates_' + setting_names[setting_code] + '_1.xlsx')
            try:
                df = pd.read_excel(file_path, sheet_name = sheet_name,header = 0)
            except:
                file_path = file_path.replace('_1.xlsx','_2.xlsx')
                df = pd.read_excel(file_path, sheet_name = sheet_name, header = None)
            return np.array(df)
        else:
            raise NotImplementedError("Invalid setting code. Try again.")
    else:
        try:
            df = pd.read_csv(file_path, delimiter = delimiter, header = header)
            return np.array(df)
        except:
            raise NotImplementedError("Contact matrix did not open. Check inputs.")


def get_contact_matrix_dic(datadir,setting_code,sheet_name=None,file_path_dic=None,delimiter = ' ',header = None):
    """
    Return a dict of setting specific age mixing matrices.
    """
    matrix_dic = {}
    if file_path_dic is None:
        file_path_dic = dict.fromkeys(['H','S','W','C'],None)
    for setting_code in ['H','S','W','C']:
        matrix_dic[setting_code] = get_contact_matrix(datadir,setting_code,sheet_name,file_path_dic[setting_code],delimiter,header)
    return matrix_dic


def combine_matrices(matrix_dic,weights_dic,num_agebrackets):
    """
    Returns a contact matrix that is a linear combination of setting specific matrices given weights for each setting.
    """
    M = np.zeros((num_agebrackets,num_agebrackets))
    for setting_code in weights_dic:
        M += matrix_dic[setting_code] * weights_dic[setting_code]
    return M


def get_ids_by_age_dic(age_by_id_dic):
    """
    Returns a dictionary listing out ids for each age from a dictionary that maps id to age.
    """
    ids_by_age_dic = {}
    for i in age_by_id_dic:
        ids_by_age_dic.setdefault( age_by_id_dic[i], [])
        ids_by_age_dic[ age_by_id_dic[i] ].append(i)
    return ids_by_age_dic


def get_uids_by_age_dic(popdict):
    """
    Returns a dictionary listing out uids for each age from a dictionary that maps uid to age.
    """
    uids_by_age_dic = {}
    for uid in popdict:
        uids_by_age_dic.setdefault( popdict[uid]['age'], [])
        uids_by_age_dic[ popdict[uid]['age'] ].append(uid)
    return uids_by_age_dic


def sample_single(distr):
    """
    Return a single sampled value from a distribution.
    """
    if type(distr) == dict:
        distr = norm_dic(distr)
        sorted_keys = sorted(distr.keys())
        sorted_distr = [distr[k] for k in sorted_keys]
        n = np.random.multinomial(1,sorted_distr,size = 1)[0]
        index = np.where(n)[0][0]
        return sorted_keys[index]
    elif type(distr) == np.ndarray:
        distr = distr / np.sum(distr)
        n = np.random.multinomial(1,distr,size = 1)[0]
        index = np.where(n)[0][0]
        return index


def resample_age(single_year_age_distr,age):
    """
    Resample age from single year age distribution.
    """
    if age == 0:
        age_min = 0
        age_max = 1
    elif age == 1:
        age_min = 0
        age_max = 2
    elif age >= 2 and age <= 98:
        age_min = age - 2
        age_max = age + 2
    elif age == 99:
        age_min = 97
        age_max = 99
    else:
        age_min = 98
        age_max = 100
    
    age_distr = norm_age_group(single_year_age_distr,age_min,age_max)
    n = np.random.multinomial(1,[age_distr[a] for a in range(age_min,age_max+1)],size = 1)[0]
    age_range = np.arange(age_min,age_max+1)
    index = np.where(n)[0]
    return age_range[index][0]


def sample_from_range(distr,min_val,max_val):
    """
    Return a sampled number from the range min_val to max_val in the distribution distr.
    """
    new_distr = norm_age_group(distr,min_val,max_val)
    return sample_single(new_distr)


def sample_bracket(distr,brackets):
    """
    Return a sampled bracket from a distribution.
    """
    if type(distr) == dict:
        sorted_keys = sorted(distr.keys())
        sorted_distr = [distr[k] for k in sorted_keys]
        n = np.random.multinomial(1,sorted_distr, size = 1)[0]
        index = np.where(n)[0][0]
    # elif type(distr) == np.ndarray:
        # distr = distr / np.sum(distr)
        # n = np.random.multinomial(1,distr,size = 1)[0]
        # index = np.where(n)[0][0]
        # 
    return index


def sample_n(nk,distr):
    """
    Return count for n samples from a distribution
    """
    if type(distr) == dict:
        distr = norm_dic(distr)
        sorted_keys = sorted(distr.keys())
        sorted_distr = [distr[k] for k in sorted_keys]
        n = np.random.multinomial(nk,sorted_distr,size = 1)[0]
        dic = dict(zip(sorted_keys,n))
        return dic
    elif type(distr) == np.ndarray:
        distr = distr / np.sum(distr)
        n = np.random.multinomial(nk, distr, size = 1)[0]
        dic = dict(zip(np.arange(len(distr)), n))
        return dic


def sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix,single_year_age_distr=None):
    """
    Return age of contact by age of individual sampled from an age mixing matrix.
    Age of contact is uniformly drawn from the age bracket sampled from the age mixing matrix.
    """
    b = age_by_brackets_dic[age]
    b_contact = sample_single(age_mixing_matrix[b,:])
    if single_year_age_distr is None:
        a = np.random.choice(age_brackets[b_contact])
    else:
        a = sample_from_range(single_year_age_distr,age_brackets[b_contact][0],age_brackets[b_contact][-1])

    return a


def sample_n_contact_ages(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic,single_year_age_distr=None):
    """
    Return n_contacts sampled from an age mixing matrix. Combines setting specific weights to create an age mixing matrix
    from which contact ages are sampled.

    For school closures or other social distancing methods, reduce n_contacts and the weights of settings that should be affected.
    """
    num_agebrackets = len(age_brackets)
    age_mixing_matrix = combine_matrices(age_mixing_matrix_dic,weights_dic,num_agebrackets)
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append( sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix,single_year_age_distr) )
    return contact_ages


def sample_n_contact_ages_with_matrix(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix,single_year_age_distr=None):
    """
    Return n_contacts sampled from an age mixing matrix.
    """
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append( sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix,single_year_age_distr) )
    return contact_ages


def get_n_contact_ids_by_age(contact_ids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic):
    """
    Return ids of n_contacts sampled from an age mixing matrix, where potential contacts are chosen from a list of contact ids by age
    """
    contact_ids = set()
    for contact_age in contact_ages:
        age_list = sorted(list(contact_ids_by_age_dic.keys()))
        ind = sc.findnearest(age_list, contact_age)
        these_ids = contact_ids_by_age_dic[age_list[ind]]
        if len(these_ids) > 0:
            contact_id = np.random.choice(these_ids)
        else:
            b_contact = age_by_brackets_dic[contact_age]
            potential_contacts = []
            for a in age_brackets[b_contact]:
                potential_contacts += contact_ids_by_age_dic[a]
            contact_id = np.random.choice( potential_contacts )
        contact_ids.add(contact_id)
    return contact_ids


@nb.njit((nb.int64,))
def pt(rate):
    ''' A Poisson trial '''
    return np.random.poisson(rate, 1)[0]


def get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets,min_age=0, max_age=100, age_mean=40, age_std=20):
    '''
    Return person's age and sex based on gender and age census data defined for age brackets. Else, return random age and sex.
    '''
    try:
        b = sample_bracket(age_bracket_distr,age_brackets)
        age = np.random.choice(age_brackets[b])
        sex = np.random.binomial(1, gender_fraction_by_age['male'][b])
        return age, sex
    except:
        sex = np.random.randint(2) # Define female (0) or male (1) -- evenly distributed
        age = np.random.normal(age_mean, age_std) # Define age distribution for the crew and guests
        age = np.median([min_age, age, max_age]) # Bound age by the interval
        return age, sex


def get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people=1,min_age=0, max_age = 100, age_mean = 40, age_std=20):
    """
    Return n_people age and sex sampled from gender and age census data defined for age brackets. Else, return random ages and sex.
    Two lists ordered by age bracket so that people from the first age bracket show up at the front of both lists and people from the last age bracket show up at the end.
    """
    n_people = int(n_people)

    if age_bracket_distr is None:
        sexes = np.random.binomial(1,p = 0.5,size = n_people)
        # ages = np.random.randint(min_age,max_age+1,size = n_people) # should return a flat distribution if we don't know the age distribution, not a normal distribution...
        ages = np.random.normal(age_mean,age_std,size = n_people)
        # ages = [a for a in ages if a >= 0]
        ages = [np.median([min_age, int(a), max_age]) for a in ages]

    else:
        bracket_count = sample_n(n_people,age_bracket_distr)
        ages, sexes = [], []

        for b in bracket_count:
            sex_probabilities = [gender_fraction_by_age['female'][b], gender_fraction_by_age['male'][b]]
            ages_in_bracket = np.random.choice(age_brackets[b],bracket_count[b])
            sexes_in_bracket = np.random.choice(np.arange(2),bracket_count[b],p = sex_probabilities)
            ages += list(ages_in_bracket)
            sexes += list(sexes_in_bracket)

    return ages, sexes


def get_seattle_age_sex(datadir,location='seattle_metro', state_location='Washington'):
    '''
    Define default age and sex distributions for Seattle
    '''
    age_bracket_distr = read_age_bracket_distr(datadir, location, state_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(datadir, census_location)
    age_brackets_file_path = get_census_age_brackets_path(datadir)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)

    age,sex = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    return age,sex


def get_seattle_age_sex_n(census_location='seattle_metro',location='Washington',n_people=1e4):
    '''
    Define default age and sex distributions for Seattle
    '''
    dropbox_path = datadir
    age_bracket_distr = read_age_bracket_distr(dropbox_path, census_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path, census_location)
    age_brackets_file_path = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)

    ages,sexes = get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people)
    return ages,sexes


def get_usa_age_sex(location='seattle_metro', state_location='Washington'):
    '''
    Define default age and sex distributions for Seattle
    '''
    dropbox_path = datadir
    country_location = 'usa'
    age_bracket_distr = read_age_bracket_distr(dropbox_path,location,state_location,country_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_file_path = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)

    age,sex = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    return age,sex


def get_usa_age_sex_n(location='seattle_metro',state_location='Washington',n_people=1e4):
    """
    Define default age and sex distributions for any place in the United States.
    """
    dropbox_path = datadir
    country_location = 'usa'
    age_bracket_distr = read_age_bracket_distr(dropbox_path,location,state_location,country_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_file_path = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)

    ages,sexes = get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people)
    return ages,sexes


def get_usa_age_n(sexes,location='seattle_metro',state_location='Washington'):
    """
    Define ages, maybe from supplied sexes for any place in the United States.
    """
    dropbox_path = datadir
    country_location = 'usa'
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_file_path = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)

    sex_count = Counter(sexes)
    sex_age_distr = {0: gender_fraction_by_age['female'], 1: gender_fraction_by_age['male']}

    ages,sexes = [],[]

    for sex in sex_count:
        bracket_count = sample_n(sex_count[sex],sex_age_distr[sex])
        for b in bracket_count:
            ages_in_bracket = np.random.choice(age_brackets[b],bracket_count[b])
            ages += list(ages_in_bracket)
        sexes += [sex] * sex_count[sex]

    return ages,sexes


def get_usa_sex_n(ages,location='seattle_metro',state_location='Washington'):
    """
    Define sexes from supplied ages for any place in the United States.
    """
    dropbox_path = datadir
    country_location = 'usa'
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_file_path = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_file_path)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)

    age_count = Counter(ages)
    bracket_count = get_aggregate_ages(age_count,age_by_brackets_dic,len(age_brackets))

    ages,sexes = [],[]

    for b in bracket_count:

        sex_probabilities = [gender_fraction_by_age['female'][b], gender_fraction_by_age['male'][b]]
        sexes_in_bracket = np.random.binomial(1,p = gender_fraction_by_age['female'][b], size = bracket_count[b])
        # sexes_in_bracket = np.random.choice(np.arange(2), bracket_count[b], sex_probabilities)
        ages_in_bracket = []
        for a in age_brackets[b]:
            ages_in_bracket += [a] * age_count[a]
        ages += ages_in_bracket
        sexes += list(sexes_in_bracket)

    return ages,sexes


def get_age_n(n,location='seattle_metro',state_location='Washington',country_location='usa',use_bayesian=False):
    """
    Define ages, regardless of sex. For webapp, set use_bayesian to True so that ages are drawn from the age brackets that match the contact matrices used by the webapp.
    """
    dropbox_path = datadir
    age_brackets = get_census_age_brackets(dropbox_path,country_location,use_bayesian)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)
    age_bracket_distr = read_age_bracket_distr(datadir,location,state_location,country_location,use_bayesian)

    ages = []
    bracket_count = sample_n(n,age_bracket_distr)
    for b in bracket_count:
        ages_in_bracket = np.random.choice(age_brackets[b],bracket_count[b])
        ages += list(ages_in_bracket)

    return ages


def get_mortality_rates_filepath(path):
    """ Return the filepath to mortality rates by age brackets. """
    return os.path.join(path,'mortality_rates_by_age_bracket.dat')


def get_mortality_rates_by_age_bracket(file_path):
    """ Return mortality rates by age brackets """
    df = pd.read_csv(file_path)
    return dict(zip(df.age_bracket,df.rate))


def get_mortality_rates_by_age(mortality_rate_by_age_bracket,mortality_age_brackets):
    """
    Return mortality rates by individual age.
    """
    mortality_rates = {}
    for b in mortality_rate_by_age_bracket:
        for a in mortality_age_brackets[b]:
            mortality_rates[a] = mortality_rate_by_age_bracket[b]
    return mortality_rates


def calc_death(person_age,mortality_rates):
    """ Return mortality rate given a person's age. """
    # return np.random.binomial(1,mortality_rates[person_age])
    return mortality_rates[person_age]
