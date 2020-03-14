import os
import synthpops as sp
import sciris as sc

import pandas as pd
import numpy as np


def test_all():
    ''' Run all tests '''
    
    sc.heading('Running all tests')

    sp.validate() # Validate that data files can be found
    dropbox_path = sp.datadir

    census_location = 'seattle_metro' # for census distributions
    location = 'Washington' # for state wide age mixing patterns

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path, census_location)

    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path, census_location)

    age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)


    ### Test selecting an age and sex for an individual ###
    a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    # a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)

    print(a,s)


    ### Test age mixing matrix ###
    num_agebrackets = 18

    # flu-like weights. calibrated to empirical diary survey data.
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79}

    age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path,location,num_agebrackets)
    age_mixing_matrix_dic['M'] = sp.get_contact_matrix(dropbox_path,location,'M',num_agebrackets)



    ### Test sampling contacts based on age ###

    # sample an age (and sex) from the seattle mCombinetro distribution
    # for n in range(10000):
        # age, sex = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)
    age, sex = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    # age, sex = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_by_brackets_dic,age_brackets)



    n_contacts = 30
    contact_ages = sp.sample_n_contact_ages(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic)
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

    contact_ages = sp.sample_n_contact_ages(n_reduced_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,no_schools_weights)

    print(contact_ages)
    



    ### mortality rates and associated functions ###
    mortality_rates_path = sp.get_mortality_rates_filepath(dropbox_path) 
    mortality_rates_by_age_bracket = sp.get_mortality_rates_by_age_bracket(mortality_rates_path)
    mortality_rate_age_brackets = sp.get_age_brackets_from_df(os.path.join(dropbox_path,'mortality_age_brackets.dat'))

    mortality_rates = sp.get_mortality_rates_by_age(mortality_rates_by_age_bracket,mortality_rate_age_brackets)
    age = 80
    prob_of_death = sp.calc_death(age,mortality_rates)
    print(prob_of_death)

    return

def test_n_single_ages(n_people=1e4):
    
    sc.heading('Running single ages')
    
    sp.validate()
    dropbox_path = sp.datadir

    census_location = 'seattle_metro' # for census distributions
    # location = 'Washington' # for state wide age mixing patterns

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path, census_location)

    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path, census_location)

    age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    # age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ### Test selecting an age and sex for an individual ###
    a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    print(a,s)

    n_people = int(n_people)

    ages, sexes = [], []
    for p in range(n_people):
        a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
        ages.append(a)
        sexes.append(s)
        # print(a,s)
    return


def test_multiple_ages(n_people=1e4):
    sc.heading('Running multiple ages')

    dropbox_path = sp.datadir

    census_location = 'seattle_metro'
    # location = 'Washington'

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path, census_location)

    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path, census_location)

    age_brackets_filepath = os.path.join(dropbox_path,'census','age distributions','census_age_brackets.dat')
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)

    ages, sexes = sp.get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people)
    # for x,y in zip(ages,sexes):
        # print(x,y)
    print(len(ages),len(sexes))

    return

def read_age_bracket_distr(datadir,location):

    f = sp.get_age_bracket_distr_path(datadir, location)
    df = pd.read_csv(f)
    return dict(zip(np.arange(len(df)),df.percent))


#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    test_all()
    test_n_single_ages(n_people=1e4)
    test_multiple_ages(n_people=1e4)
    sc.toc()





print('Done.')