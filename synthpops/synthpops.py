import os
import numpy as np
import pandas as pd
import sciris as sc
import numba  as nb
from collections import Counter
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


def get_gender_fraction_by_age_path(datadir, location, state_location=None, country_location=None):
    """
    Return filepath for all Seattle Metro gender fractions by age bracket.
    """
    if state_location == None:
        return os.path.join(datadir,'census','age distributions',location + '_gender_fraction_by_age_bracket.dat')
    else:
        return os.path.join(datadir,'demographics',country_location,state_location,'census','age distributions',location + '_gender_fraction_by_age_bracket.dat')


def read_gender_fraction_by_age_bracket(datadir, location, state_location=None, country_location=None):
    """
    Return dict of gender fractions by age bracket for all Seattle Metro.
    """
    if state_location == None:
        f = get_gender_fraction_by_age_path(datadir, location)
    else:
        f = get_gender_fraction_by_age_path(datadir, location, state_location, country_location)
    df = pd.read_csv(f)
    dic = {}
    dic['male'] = dict(zip(np.arange(len(df)),df.fraction_male))
    dic['female'] = dict(zip(np.arange(len(df)),df.fraction_female))
    return dic


def get_age_bracket_distr_path(datadir, location, state_location=None, country_location=None,use_bayesian=False):
    """
    Return filepath for age distribution by age brackets.
    For the webapp, if using in combination with age-mixing contact matrices, set use_bayesian to True. Otherwise, there are no associated contact matrices to use and you'll need to use a generic contact network later on.
    """
    if use_bayesian:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions',location + '_age_bracket_distr.dat')
    else:
        if state_location == None:
            return os.path.join(datadir,'census','age distributions',location + '_age_bracket_distr.dat')
        else:
            return os.path.join(datadir,'demographics',country_location,state_location,'census','age distributions',location + '_age_bracket_distr.dat')


def read_age_bracket_distr(datadir, location, state_location=None, country_location=None,use_bayesian=False):
    """
    Return dict of age distribution by age brackets.
    For the webapp, if using in combination with age-mixing contact matrices set use_bayesian to True. Otherwise, there are no associated contact matrices to use and you'll need to use a generic contact network later on.
    """
    df = pd.read_csv(get_age_bracket_distr_path(datadir,location,state_location,country_location,use_bayesian))
    return dict(zip(np.arange(len(df)), df.percent))


def get_age_brackets_from_df(ab_filepath):
    """
    Returns dict of age bracket ranges from ab_filepath.
    """
    ab_df = pd.read_csv(ab_filepath,header = None)
    dic = {}
    for index,row in enumerate(ab_df.iterrows()):
        age_min = row[1].values[0]
        age_max = row[1].values[1]
        dic[index] = np.arange(age_min,age_max+1)
    return dic


def get_census_age_brackets_path(datadir,country_location=None,use_bayesian=False):
    """
    Returns filepath for census age brackets : depends on the country or source of contact pattern data.
    For the webapp where we currently use the contact matrices from K. Prem et al, set flag use_bayesian to True.
    """
    if use_bayesian:
        return os.path.join(datadir,'demographics','contact_matrices_152_countries','bayesian_152_countries_age_brackets.dat')
    else:
        if country_location == None:
            return os.path.join(datadir,'demographics','generic_census_age_brackets.dat')
        elif country_location == 'usa':
            return os.path.join(datadir,'demographics',country_location,'census_age_brackets.dat')


def get_census_age_brackets(datadir,country_location=None,use_bayesian=False):
    """
    Returns census age brackets: depends on the country or source of contact patterb data.
    For the webapp where we currently use the contact matrices from K. Prem et al, set flag use_bayesian to True.
    """
    if use_bayesian:
        filepath = os.path.join(datadir,'demographics','contact_matrices_152_countries','bayesian_152_countries_age_brackets.dat')
    else:
        if country_location == None:
            filepath = os.path.join(datadir,'demographics','generic_census_age_brackets.dat')
        elif country_location == 'usa':
            filepath = os.path.join(datadir,'demographics',country_location,'census_age_brackets.dat')

    return get_age_brackets_from_df(filepath)


def get_age_by_brackets_dic(age_brackets):
    """
    Returns dict of age bracket by age.
    """
    age_by_brackets_dic = {}
    for b in age_brackets:
        for a in age_brackets[b]:
            age_by_brackets_dic[a] = b
    return age_by_brackets_dic


def get_aggregate_ages(ages,age_by_brackets_dic,num_agebrackets):
    """
    Return an aggregate age count for specified age brackets (values in age_by_brackets_dic)
    """
    bracket_keys = set(age_by_brackets_dic.values())
    aggregate_ages = dict.fromkeys(bracket_keys,0)
    for a in ages:
        b = age_by_brackets_dic[a]
        aggregate_ages[b] += ages[a]
    return aggregate_ages


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
    census_age_bracket_distr = read_age_bracket_distr(datadir,location,state_location,country_location)
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
    Return symmetric homogeneous community matrix for age count in ages
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


def get_contact_matrix(datadir,state_location,setting_code,num_agebrackets=None,use_bayesian=False,sheet_name='United States of America'):
    """
    Return setting specific contact matrix for num_agebrackets age brackets.
    For setting code M, returns an influenza weighted combination of the settings: H, S, W, R.
    """

    # notably this uses a different definition of the community matrix : has assortative age structure
    if use_bayesian:
        setting_names = {'H': 'home', 'S': 'school', 'W' : 'work', 'R': 'other_locations'}
        if setting_code in ['H','S','W','R']:
            filepath = os.path.join(datadir,'demographics','contact_matrices_152_countries','MUestimates_' + setting_names[setting_code] + '_1.xlsx')
            try:
                df = pd.read_excel(filepath, sheet_name = sheet_name,header = 0)
            except:
                filepath = filepath.replace('_1.xlsx','_2.xlsx')
                df = pd.read_excel(filepath, sheet_name = sheet_name,header = None)
            return np.array(df)
        # elif setting_code == 'R':
            # age_bracket_distr = sp.read_age_bracket_distr(datadir,location,state_location,country_location)
        else: raise NotImplementedError('Setting contact matrix does not exist.')
    else:
        file_path = os.path.join(datadir,'SyntheticPopulations','asymmetric_matrices','data_' + setting_code + str(num_agebrackets),'M' + str(num_agebrackets) + '_' + state_location + '_' + setting_code + '.dat')
        return np.array(pd.read_csv(file_path,delimiter = ' ', header = None))


def get_contact_matrix_dic(datadir, state_location, num_agebrackets=None,use_bayesian=False,sheet_name='United States of America'):
    """
    Return a dict of setting specific age mixing matrices.
    """
    matrix_dic = {}
    for setting_code in ['H','S','W','R']:
        matrix_dic[setting_code] = get_contact_matrix(datadir,state_location,setting_code,num_agebrackets,use_bayesian,sheet_name)
    return matrix_dic


def combine_matrices(matrix_dic,weights_dic,num_agebrackets):
    """
    Returns a contact matrix that is a linear combination of setting specific matrices given weights for each setting.
    """
    M = np.zeros((num_agebrackets,num_agebrackets))
    for setting_code in weights_dic:
        M += matrix_dic[setting_code] * weights_dic[setting_code]
    return M


def get_ages(synpop_path,location,num_agebrackets):
    """
    Return synthetic age counts for num_agebrackets age brackets.
    """
    file_path = os.path.join(datadir,'SyntheticPopulations','synthetic_ages','data_a' + str(num_agebrackets),'a' + str(num_agebrackets) + '_' + location + '.dat')
    df = pd.read_csv(file_path, delimiter = ' ', header = None)
    return dict(zip(df.iloc[:,0].values, df.iloc[:,1].values))


def get_ids_by_age_dic(age_by_id_dic):
    """
    Returns a dictionary listing out ids for each age
    """
    ids_by_age_dic = {}
    for i in age_by_id_dic:
        ids_by_age_dic.setdefault( age_by_id_dic[i], [])
        ids_by_age_dic[ age_by_id_dic[i] ].append(i)
    return ids_by_age_dic


def get_uids_by_age_dic(popdict):
    """
    Returns a dictionary listing out uids for each age
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


def sample_bracket(distr,brackets):
    """
    Return a sampled bracket from a distribution.
    """
    sorted_keys = sorted(distr.keys())
    sorted_distr = [distr[k] for k in sorted_keys]
    n = np.random.multinomial(1,sorted_distr, size = 1)[0]
    index = np.where(n)[0][0]
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

def sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix):
    """
    Return age of contact by age of individual sampled from an age mixing matrix.
    Age of contact is uniformly drawn from the age bracket sampled from the age mixing matrix.
    """
    b = age_by_brackets_dic[age]
    b_contact = sample_single(age_mixing_matrix[b,:])
    return np.random.choice(age_brackets[b_contact])


def sample_n_contact_ages(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic,num_agebrackets=18):
    """
    Return n_contacts sampled from an age mixing matrix. Combines setting specific weights to create an age mixing matrix
    from which contact ages are sampled.

    For school closures or other social distancing methods, reduce n_contacts and the weights of settings that should be affected.
    """
    num_agebrackets = len(age_brackets)
    age_mixing_matrix = combine_matrices(age_mixing_matrix_dic,weights_dic,num_agebrackets)
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append( sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix) )
    return contact_ages


def sample_n_contact_ages_with_matrix(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix,num_agebrackets=18):
    """
    Return n_contacts sampled from an age mixing matrix.
    """
    contact_ages = []
    for i in range(n_contacts):
        contact_ages.append( sample_contact_age(age,age_brackets,age_by_brackets_dic,age_mixing_matrix) )
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


def get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets,min_age=0, max_age=99, age_mean=40, age_std=20):
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


def get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people=1,min_age=0, max_age = 99, age_mean = 40, age_std=20):
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


def get_seattle_age_sex(census_location='seattle_metro', location='Washington'):
    '''
    Define default age and sex distributions for Seattle
    '''
    dropbox_path = datadir
    age_bracket_distr = read_age_bracket_distr(dropbox_path, census_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path, census_location)
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)

    age,sex = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    return age,sex


def get_seattle_age_sex_n(census_location='seattle_metro',location='Washington',n_people=1e4):
    '''
    Define default age and sex distributions for Seattle
    '''
    dropbox_path = datadir
    age_bracket_distr = read_age_bracket_distr(dropbox_path, census_location)
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path, census_location)
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)

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
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)

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
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)

    ages,sexes = get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people)
    return ages,sexes


def get_usa_age_n(sexes,location='seattle_metro',state_location='Washington'):
    """
    Define ages, maybe from supplied sexes for any place in the United States.
    """
    dropbox_path = datadir
    country_location = 'usa'
    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)
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
    age_brackets_filepath = get_census_age_brackets_path(dropbox_path)
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)
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


def get_mortality_rates_by_age_bracket(filepath):
    """ Return mortality rates by age brackets """
    df = pd.read_csv(filepath)
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
