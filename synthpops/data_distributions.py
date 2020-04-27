"""
Read in data distributions.
"""

import os
import numpy as np
import pandas as pd
import sciris as sc
import numba as nb
from collections import Counter
from copy import deepcopy
from . import synthpops as sp
from .config import datadir


def get_age_brackets_from_df(ab_file_path):
    """
    Returns dict of age bracket ranges from ab_file_path.
    """
    ab_df = pd.read_csv(ab_file_path, header=None)
    dic = {}
    for index, row in enumerate(ab_df.iterrows()):
        age_min = row[1].values[0]
        age_max = row[1].values[1]
        dic[index] = np.arange(age_min, age_max+1)
    return dic


def get_gender_fraction_by_age_path(datadir, location=None, state_location=None, country_location=None):
    """
    Return file_path for gender fractions by age bracket. This should only be used if the data is available.
    """
    if location is None:
        raise NotImplementedError
    levels = [location, state_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location, state_location, and country_location strings.")
    else:
        if state_location is None:  # use this is if you want to get national data
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'age distributions', location + '_gender_fraction_by_age_bracket_16.dat')
        # if country_location is None:
            # return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions',location + '_gender_fraction_by_age_bracket_16.dat')
        else:
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age distributions', location + '_gender_fraction_by_age_bracket_16.dat')
        # return os.path.join(datadir,'demographics',country_location,state_location,'census','age distributions',location + '_gender_fraction_by_age_bracket.dat')


def read_gender_fraction_by_age_bracket(datadir, location=None, state_location=None, country_location=None, file_path=None):
    """
    Return dict of gender fractions by age bracket, either by location, state_location, country_location strings, or by the file_path if that's given.
    """
    if file_path is None:
        file_path = get_gender_fraction_by_age_path(datadir, location, state_location, country_location)
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path)
    dic = {}
    dic['male'] = dict(zip(np.arange(len(df)), df.fraction_male))
    dic['female'] = dict(zip(np.arange(len(df)), df.fraction_female))
    return dic


def get_age_bracket_distr_path(datadir, location=None, state_location=None, country_location=None):
    """
    Return file_path for age distribution by age brackets.
    """
    if location is None:
        raise NotImplementedError
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    else:
        if state_location is None:  # use this for national data.
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'age distributions', location + '_age_bracket_distr_16.dat')
        # if country_location is None:
            # return os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,'age distributions',location + '_age_bracket_distr_16.dat')
        else:
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age distributions', location + '_age_bracket_distr_16.dat')


def read_age_bracket_distr(datadir, location=None, state_location=None, country_location=None, file_path=None):
    """
    Return dict of age distribution by age brackets.
    """
    if file_path is None:
        file_path = get_age_bracket_distr_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return dict(zip(np.arange(len(df)), df.percent))


def get_household_size_distr_path(datadir, location=None, state_location=None, country_location=None):
    """
    Return file_path for household size distribution
    """
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    else:
        if state_location is None:  # use for national data
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'household size distributions', location + '_household_size_distr.dat')
        else:
            return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household size distributions', location + '_household_size_distr.dat')


def get_household_size_distr(datadir, location=None, state_location=None, country_location=None, file_path=None):
    """
    Return a dictionary of the distributions of household sizes. If you don't give the file_path, then supply the location and state_location strings.
    """
    if file_path is None:
        file_path = get_household_size_distr_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return dict(zip(df.household_size, df.percent))


def get_head_age_brackets_path(datadir, state_location=None, country_location=None):
    """
    Return file_path for head of household age brackets. If data doesn't exist at the state level, only give the country_location.
    """
    levels = [state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'household living arrangements', 'head_age_brackets.dat')
    else:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household living arrangements', 'head_age_brackets.dat')


def get_head_age_brackets(datadir, state_location=None, country_location=None, file_path=None):
    """
    Return head age brackets either from the file_path directly, or using the other parameters to figure out what the file_path should be.
    """
    if file_path is None:
        file_path = get_head_age_brackets_path(datadir, state_location, country_location)
    return get_age_brackets_from_df(file_path)


def get_household_head_age_by_size_path(datadir, state_location=None, country_location=None):
    """
    Return file_path for head of household age by size counts or distribution. If the data doesn't exist at the state level, only give the country_location.
    """
    levels = [state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'household living arrangements', 'household_head_age_and_size_count.dat')
    else:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household living arrangements', 'household_head_age_and_size_count.dat')


def get_household_head_age_by_size_df(datadir, state_location=None, country_location=None, file_path=None):
    """
    Return a pandas df of head of household age by the size of the household. If the file_path is given return from there first.
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir, state_location, country_location)
    return pd.read_csv(file_path)


def get_head_age_by_size_distr(datadir, state_location=None, country_location=None, file_path=None, household_size_1_included=False):
    """
    Return an array of head of household age bracket counts (col) given by size (row).
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir, state_location, country_location)
    hha_df = get_household_head_age_by_size_df(datadir, state_location, country_location, file_path)
    hha_by_size = np.zeros((2 + len(hha_df), len(hha_df.columns)-1))
    if household_size_1_included:
        for s in range(1, len(hha_df)+1):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    else:
        hha_by_size[0, :] += 1
        for s in range(2, len(hha_df)+2):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    return hha_by_size


def get_census_age_brackets_path(datadir, state_location=None, country_location=None):
    """
    Returns file_path for census age brackets: depends on the state or country of the source data on contact patterns.
    """
    levels = [state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'census_age_brackets.dat')
    else:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'census_age_brackets.dat')


def get_census_age_brackets(datadir, state_location=None, country_location=None, file_path=None):
    """
    Returns census age brackets: depends on the country or source of contact pattern data.
    """
    if file_path is None:
        file_path = get_census_age_brackets_path(datadir, state_location, country_location)
    return get_age_brackets_from_df(file_path)


def get_contact_matrix(datadir, setting_code, sheet_name=None, file_path=None, delimiter=' ', header=None):
    """
    Return setting specific contact matrix givn sheet name to use. If file_path is given, then delimiter and header should also be specified.
    """
    if file_path is None:
        setting_names = {'H': 'home', 'S': 'school', 'W': 'work', 'C': 'other_locations'}
        if setting_code in setting_names:
            file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', 'MUestimates_' + setting_names[setting_code] + '_1.xlsx')
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
            except:
                file_path = file_path.replace('_1.xlsx', '_2.xlsx')
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            return np.array(df)
        else:
            raise NotImplementedError("Invalid setting code. Try again.")
    else:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, header=header)
            return np.array(df)
        except:
            raise NotImplementedError("Contact matrix did not open. Check inputs.")


def get_contact_matrix_dic(datadir, sheet_name=None, file_path_dic=None, delimiter=' ', header=None):
    """
    Return a dict of setting specific age mixing matrices.
    """
    matrix_dic = {}
    if file_path_dic is None:
        file_path_dic = dict.fromkeys(['H', 'S', 'W', 'C'], None)
    for setting_code in ['H', 'S', 'W', 'C']:
        matrix_dic[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name, file_path_dic[setting_code], delimiter, header)
    return matrix_dic


# School enrollment data specific for Seattle / United States of America. Change name to reflect that

def get_usa_school_enrollment_rates_df(datadir, locations, location, state_location, country_location, level):

    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, location, 'schools', level + '_school_enrollment_by_age','ACSST5Y2018.S1401_data_with_overlays_2020-03-06T233142.csv')
    df = pd.read_csv(file_path, header=1)
    if type(locations) == list:
        d = df[df['Geographic Area Name'].isin(locations)]
    else:
        d = df[df['Geographic Area Name'] == locations]
    return d


def process_usa_school_enrollment_rates(datadir, locations, location, state_location, country_location, level):

    df = get_usa_school_enrollment_rates_df(datadir, locations, location, state_location, country_location, level)
    skip_labels = ['Error', 'public', 'private', 'X', 'Total', 'Male', 'Female', '3 years and over', '18 to 24']
    columns = df.columns
    columns = [col for col in columns if not any(l in col for l in skip_labels)]

    rates = dict.fromkeys(np.arange(101), 0)
    # process into enrollment rates by age
    for col in columns:
        if 'enrolled in school' in col:

            age_bracket = col.replace(' year olds enrolled in school', '')
            age_bracket = age_bracket.split('!!')[-1]

            if ' to ' in age_bracket:
                age_bracket = age_bracket.split(' to ')
            elif ' and ' in age_bracket and 'over' not in age_bracket:
                age_bracket = age_bracket.split(' and ')
            elif 'over' in age_bracket:
                age_bracket = age_bracket.replace(' years and over enrolled in school', '')

            if type(age_bracket) == list:
                sa, ea = int(age_bracket[0]), int(age_bracket[1])
            else:
                sa = int(age_bracket)
                ea = sa + 15  # arbitrary guess of what age ends the age bracket

            for a in np.arange(sa, ea+1):
                rates[a] = np.round(df[col].values[0]/100, 8)
    return rates


def write_school_enrollment_rates(datadir, locations, location, state_location, country_location, level):

    rates = process_usa_school_enrollment_rates(datadir, locations, location, state_location, country_location, level)
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'enrollment')
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, location + '_school_enrollment_by_age.dat')
    f = open(file_path, 'w')
    f.write('Age,Percent\n')
    for a in rates:
        f.write(str(a) + ',' + str(rates[a]) + '\n')
    f.close()


def get_school_enrollment_rates_path(datadir, location=None, state_location=None, country_location=None):
    """
    Return_path for enrollment rates by age.
    """
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None and state_location is not None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'enrollment', 'school_enrollment_by_age.dat')
    elif state_location is None:
        raise NotImplementedError("Missing state_location input string. Try again. ")
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'enrollment', location + '_school_enrollment_by_age.dat')


def get_school_enrollment_rates(datadir, location=None, state_location=None, country_location=None, file_path=None):
    """ Return dictionary of enrollment rates by age. """
    if file_path is None:
        file_path = get_school_enrollment_rates_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return dict(zip(df.Age, df.Percent))


# Generalized function for any location that has enrollment sizes

def get_school_size_brackets_path(datadir, location, state_location, country_location):
    """ Return file_path for school size brackets. """
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None and state_location is not None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'schools', 'school_size_brackets.dat')
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'schools', 'school_size_brackets.dat')
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, location, 'schools', location + '_school_size_brackets.dat')


def get_school_size_brackets(datadir, location, state_location, country_location, file_path=None):

    if file_path is None:
        file_path = get_school_size_brackets_path(datadir, location, state_location, country_location)
        # file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, location, 'schools', 'school_size_brackets.dat')
    return sp.get_age_brackets_from_df(file_path)


def get_school_sizes_path(datadir, location=None, state_location=None, country_location=None):
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'schools', 'school_sizes.dat')
    elif state_location is None:
        raise NotImplementedError("Missing state_location input string. Try again.")
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, location, 'schools', location + '_school_sizes.dat')


def get_school_sizes_df(datadir, location=None, state_location=None, country_location=None, file_path=None):
    if file_path is None:
        file_path = get_school_sizes_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return df


def get_school_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None):
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None and state_location is not None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'schools', 'school_size_distr.dat')
    elif state_location is None:
        raise NotImplementedError("Missing state_location input string. Try again.")
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, location, 'schools', location + '_school_size_distr.dat')


def get_school_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, counts_available=False, size_distr_file_path=None):
    """ Either you have enrollments by individual school or you have school size distribution that is binned. Either way, you want to get a school size distribution."""
    # create size distribution from enrollment counts
    if counts_available:
        df = get_school_sizes_df(datadir, location, state_location, country_location)
        sizes = df.iloc[:, 0].values
        size_count = Counter(sizes)

        size_brackets = get_school_size_brackets(datadir, location, state_location, country_location)  # add option to give input filenames!
        size_by_bracket_dic = sp.get_age_by_brackets_dic(size_brackets)

        bracket_count = dict.fromkeys(np.arange(len(size_brackets)), 0)

        # aggregate the counts by bracket or bins
        for s in size_count:
            b = size_by_bracket_dic[s]
            bracket_count[b] += size_count[s]

        size_distr = sp.norm_dic(bracket_count)
    # read in size distribution from data file
    else:
        if size_distr_file_path is None:
            size_distr_file_path = get_school_size_distr_by_brackets_path(datadir, location, state_location, country_location)
        df = pd.read_csv(size_distr_file_path)
        size_distr = dict(zip(df.size_bracket, df.percent))
        size_distr = sp.norm_dic(size_distr)

    return size_distr


# binning school sizes for Seattle
def get_usa_school_sizes_by_bracket(datadir, location, state_location, country_location):
    df = get_school_sizes_df(datadir, location, state_location, country_location)
    sizes = df.iloc[:, 0].values
    size_count = Counter(sizes)

    size_brackets = get_school_size_brackets(datadir, location, state_location, country_location)
    size_by_bracket_dic = sp.get_age_by_brackets_dic(size_brackets)  # not actually ages just a useful function for mapping ints back to their bracket or grouping key

    bracket_count = dict.fromkeys(np.arange(len(size_brackets)), 0)

    for s in size_count:
        bracket_count[size_by_bracket_dic[s]] += size_count[s]

    count_by_mean = {}

    for b in bracket_count:
        size = int(np.mean(size_brackets[b]))
        count_by_mean[size] = bracket_count[b]

    return count_by_mean


def get_employment_rates_path(datadir, location=None, state_location=None, country_location=None):
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None and state_location is not None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'employment', 'employment_rates_by_age.dat')
    elif state_location is None:
        raise NotImplementedError("Missing state_location input string. Try again.")
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'employment', location + '_employment_rates_by_age.dat')


def get_employment_rates(datadir, location, state_location, country_location, file_path=None):

    if file_path is None:
        file_path = get_employment_rates_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return dict(zip(df.Age, df.Percent))


# def get_employment_rates(datadir, location, state_location, country_location):
    # file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'employment', location + '_employment_pct_by_age.csv')
    # df = pd.read_csv(file_path)
    # dic = dict(zip(df.Age, df.Percent))
    # return dic


def get_workplace_size_brackets_path(datadir, location=None, state_location=None, country_location=None):
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None and state_location is not None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'workplaces', 'work_size_brackets.dat')
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'workplaces', 'work_size_brackets.dat')
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'workplaces', location + '_work_size_brackets.dat')


def get_workplace_size_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None):
    if file_path is None:
        file_path = get_workplace_size_brackets_path(datadir, location, state_location, country_location)
    return sp.get_age_brackets_from_df(file_path)


# def get_workplace_size_brackets(datadir, country_location):

    # file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'work_size_brackets.dat')
    # return sp.get_age_brackets_from_df(file_path)


def get_workplace_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None):
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'workplaces', 'work_size_count.dat')
    elif state_location is None:
        return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'workplaces', 'work_size_count.dat')
    return os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'workplaces', location + '_work_size_brackets.dat')


def get_workplace_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None):
    if file_path is None:
        file_path = get_workplace_size_distr_by_brackets_path(datadir, location, state_location, country_location)
    df = pd.read_csv(file_path)
    return dict(zip(df.work_size_bracket, df.size_count))


# def get_workplace_sizes(datadir, country_location):

    # file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, 'work_size_count.dat')
    # df = pd.read_csv(file_path)
    # return dict(zip(df.work_size_bracket, df.size_count))
