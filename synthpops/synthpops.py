import os
import numpy as np
import pylab as pl
import pandas as pd
import sciris as sc

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

def get_gender_fraction_by_age_path(dropbox_path,location):
    """ Return filepath for all Seattle Metro gender fractions by age bracket. """
    return os.path.join(dropbox_path,'census','age distributions',location + '_gender_fraction_by_age_bracket.dat')

def read_gender_fraction_by_age_bracket(dropbox_path,location):
    """ 
    Return dict of gender fractions by age bracket for all Seattle Metro.

    """
    f = get_gender_fraction_by_age_path(dropbox_path,location)
    df = pd.read_csv(f)
    dic = {}
    dic['male'] = dict(zip(np.arange(len(df)),df.fraction_male))
    dic['female'] = dict(zip(np.arange(len(df)),df.fraction_female))
    return dic

def get_age_bracket_distr_path(dropbox_path,location):
    """ Return filepath for age distribution by age brackets. """
    return os.path.join(dropbox_path,'census','age distributions',location + '_age_bracket_distr.dat')


def read_age_bracket_distr(dropbox_path,location):
    """
    Return dict of age distribution by age brackets. 

    """

    f = get_age_bracket_distr_path(dropbox_path,location)
    df = pd.read_csv(f)
    return dict(zip(df.age_bracket,df.percent))

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

def get_age_by_brackets_dic(age_brackets):
    """
    Returns dict of age bracket by age.
    """
    age_by_brackets_dic = {}
    for b in age_brackets:
        for a in age_brackets[b]:
            age_by_brackets_dic[a] = b
    return age_by_brackets_dic

def get_contact_matrix(synpop_path,location,setting_code,num_agebrackets):
    """
    Return setting specific contact matrix for num_agebrackets age brackets. 
    For setting code M, returns an influenza weighted combination of the settings: H, S, W, R.
    """
    file_path = os.path.join(dropbox_path,'SyntheticPopulations','asymmetric_matrices','data_' + setting_code + str(num_agebrackets),'M' + str(num_agebrackets) + '_' + location + '_' + setting_code + '.dat')
    return np.array(pd.read_csv(file_path,delimiter = ' ', header = None))

def get_contact_matrix_dic(dropbox_path,location,num_agebrackets):
    """
    Return a dict of setting specific age mixing matrices. 
    """
    matrix_dic = {}
    for setting_code in ['H','S','W','R']:
        matrix_dic[setting_code] = get_contact_matrix(dropbox_path,location,setting_code,num_agebrackets)
    return matrix_dic

def combine_matrices(matrix_dic,weights_dic,num_agebrackets):
    M = np.zeros((num_agebrackets,num_agebrackets))
    for setting_code in weights_dic:
        M += matrix_dic[setting_code] * weights_dic[setting_code]
    return M

def get_ages(synpop_path,location,num_agebrackets):
    """
    Return synthetic age counts for num_agebrackets age brackets.
    """
    file_path = os.path.join(dropbox_path,'SyntheticPopulations','synthetic_ages','data_a85','a85_' + location + '.dat')
    df = pd.read_csv(file_path, delimiter = ' ', header = None)
    return dict(zip(df.iloc[:,0].values, df.iloc[:,1].values))

def get_ids_by_age_dic(age_by_id_dic):
    ids_by_age_dic = {}
    for i in age_by_id_dic:
        ids_by_age_dic.setdefault( age_by_id_dic[i], [])
        ids_by_age_dic[ age_by_id_dic[i] ].append(i)
    return ids_by_age_dic

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
    age_mixing_matrix = combine_matrices(age_mixing_matrix_dic,weights_dic,num_agebrackets)
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
        if len(contact_ids_by_age_dic[contact_age]) > 0:
            contact_id = np.choice( contact_ids_by_age_dic[contact_age] )
        else:
            b_contact = age_by_brackets_dic[contact_age]
            potential_contacts = []
            for a in age_brackets[b_contact]:
                potential_contacts += contact_ids_by_age_dic[a]
            contact_id = np.choice( potential_contacts )
        contact_ids.add(contact_id)
    return contact_ids

def get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets,age_brackets,min_age=0, max_age=99, age_mean=40, age_std=20):
    '''
    Define age-sex distributions.
     
    '''
    try:
        b = sample_bracket(age_bracket_distr,age_brackets)
        age = np.random.choice(age_brackets[b])
        sex = np.random.binomial(1, gender_fraction_by_age['male'][b])
        return age, sex
    except:
        sex = pl.randint(2) # Define female (0) or male (1) -- evenly distributed
        age = pl.normal(age_mean, age_std) # Define age distribution for the crew and guests
        age = pl.median([min_age, age, max_age]) # Normalize
        return age, sex




def get_mortality_rates_filepath(path):
    return os.path.join(path,'mortality_rates_by_age_bracket.dat')

def get_mortality_rates_by_age_bracket(filepath):
    df = pd.read_csv(filepath)
    return dict(zip(df.age_bracket,df.rate))

def get_mortality_rates_by_age(mortality_rate_by_age_bracket,mortality_age_brackets):
    mortality_rates = {}
    for b in mortality_rate_by_age_bracket:
        for a in mortality_age_brackets[b]:
            mortality_rates[a] = mortality_rate_by_age_bracket[b]
    return mortality_rates

def calc_death(person_age,mortality_rates):
    # return np.random.binomial(1,mortality_rates[person_age])
    return mortality_rates[person_age]


if __name__ == "__main__":

    dropbox_path = os.path.join('/home','dmistry','Dropbox (IDM)','seattle_network') # shared dropbox for seattle_network
    # synpop_path = os.path.join('/home','dmistry','Dropbox (IDM)','dmistry_COVID-19') # wherever you put the private age mixing data from Dina's Dropbox link. Please read README.md and do not put in other shared folders.

    
    census_location = 'seattle_metro' # for census distributions
    location = 'Washington' # for state wide age mixing patterns

    age_bracket_distr = read_age_bracket_distr(dropbox_path,census_location)

    gender_fraction_by_age = read_gender_fraction_by_age_bracket(dropbox_path,census_location)

    age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
    age_brackets = get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)


    ### Test selecting an age and sex for an individual ###
    a,s = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)
    print(a,s)



    ### Test age mixing matrix ###
    num_agebrackets = 18
    setting_codes = ['H','S','W','R']

    # flu-like weights. calibrated to empirical diary survey data.
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79}

    age_mixing_matrix_dic = get_contact_matrix_dic(dropbox_path,location,num_agebrackets)
    age_mixing_matrix_dic['M'] = get_contact_matrix(dropbox_path,location,'M',num_agebrackets)



    ### Test sampling contacts based on age ###

    # sample an age (and sex) from the seattle mCombinetro distribution
    age, sex = get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)

    n_contacts = 30
    contact_ages = sample_n_contact_ages(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic)
    print(contact_ages)

    # shut down schools

    no_schools_weights = sc.dcp(weights_dic)
    no_schools_weights['S'] = 0.1 # research shows that even with school closure, kids still have some contact with their friends from school.

    f_reduced_contacts_students = 0.5
    f_reduced_contacts_nonstudents = 0.2

    if age < 20: 
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_students))
    else:
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_nonstudents))

    contact_ages = sample_n_contact_ages(n_reduced_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,no_schools_weights)

    print(contact_ages)



    ### mortality rates and associated functions ###
    mortality_rates_path = get_mortality_rates_filepath(dropbox_path) 
    mortality_rates_by_age_bracket = get_mortality_rates_by_age_bracket(mortality_rates_path)
    mortality_rate_age_brackets = get_age_brackets_from_df(os.path.join(dropbox_path,'mortality_age_brackets.dat'))

    mortality_rates = get_mortality_rates_by_age(mortality_rates_by_age_bracket,mortality_rate_age_brackets)
    age = 80
    prob_of_death = calc_death(age,mortality_rates)
    print(prob_of_death)
