import numpy as np
import pandas as pd
import synthpops as sp
from glob import glob as ls
import os


def process_census_age_counts(datadir, location, state_location, country_location, year, acs_period):
    """
    Process American Community Survey data for a given year to get an age count for the location binned into 18 age brackets.

    Args:
        datadir (str)          : file path to the data directory
        location (str)         : name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        year (int)             : the year for the American Community Survey
        acs_period (int)       : the number of years for the American Community Survey

    Returns:
        A dictionary with the binned age count and a dictionary with the age bracket ranges.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_path = ls(os.path.join(file_path, 'ACSST{acs_period}Y{year}.S0101_data_with_overlays_{location}*'.format(acs_period=acs_period, year=year, location=location)))[0]

    df = pd.read_csv(file_path)

    columns = ['S0101_C01_00' + '%i' % i + 'E' for i in range(2, 10)] + ['S0101_C01_0' + '%i' % i + 'E' for i in range(10, 20)]

    age_brackets = {}
    for b in np.arange(0, len(columns) - 1):
        age_brackets[b] = np.arange(5 * b, 5 * (b + 1))
    age_brackets[len(age_brackets)] = np.arange(5 * len(age_brackets), 101)

    age_bracket_count = {}
    for b in age_brackets:
        c = columns[b]
        count = int(df.loc[df['NAME'] == location][c].values[0])
        age_bracket_count[b] = count

    return age_bracket_count, age_brackets


def process_census_age_counts_by_gender(datadir, location, state_location, country_location, year, acs_period):
    """
    Process American Community Survey data for a given year to get an age count by gender for the location binned into 18 age brackets.

    Args:
        datadir (str)          : file path to the data directory
        location (str)         : name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        year (int)             : the year for the American Community Survey
        acs_period (int)       : the number of years for the American Community Survey

    Returns:
        A dictionary with the binned age count by gender and a dictionary with the age bracket ranges.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_path = ls(os.path.join(file_path, 'ACSST{acs_period}Y{year}.S0101_data_with_overlays_{location}*'.format(acs_period=acs_period, year=year, location=location)))[0]

    df = pd.read_csv(file_path)

    columns_male = ['S0101_C03_00' + '%i' % i + 'E' for i in range(2, 10)] + ['S0101_C03_0' + '%i' % i + 'E' for i in range(10, 20)]
    columns_female = [col.replace('C03', 'C05') for col in columns_male]
    age_brackets = {}
    for b in range(0, len(columns_male) - 1):
        age_brackets[b] = np.arange(5 * b, 5 * (b + 1))
    age_brackets[len(age_brackets)] = np.arange(5 * len(age_brackets), 101)

    age_bracket_count_by_gender = {g: {} for g in ['male', 'female']}
    for b in age_brackets:
        mc = columns_male[b]
        fc = columns_female[b]
        mcount = int(df.loc[df['NAME'] == location][mc].values[0])
        fcount = int(df.loc[df['NAME'] == location][fc].values[0])
        age_bracket_count_by_gender['male'][b] = mcount
        age_bracket_count_by_gender['female'][b] = fcount

    return age_bracket_count_by_gender, age_brackets


def process_census_household_size_count(datadir, location, state_location, country_location, year, acs_period):
    """
    Process American Community Survey data for a given year to get a household size count for the location. The last bin represents households of size 7 or higher.

    Args:
        datadir (str)          : file path to the data directory
        location (str)         : name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        year (int)             : the year for the American Community Survey
        acs_period (int)       : the number of years for the American Community Survey

    Returns:
        A dictionary with the household size count.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household_size_distributions')
    file_path = ls(os.path.join(file_path, 'ACSDT{acs_period}Y{year}.B11016_data_with_overlays_{location}*'.format(acs_period=acs_period, year=year, location=location)))[0]

    df = pd.read_csv(file_path)
    columns_family = ['B11016_00' + '%i' % i + 'E' for i in range(3, 9)]
    columns_nonfamily = ['B11016_0' + '%i' % i + 'E' for i in range(10, 17)]
    household_size_count = dict.fromkeys(np.arange(1, 8), 0)
    household_size_count[1] = int(df.loc[df['NAME'] == location]['B11016_010E'].values[0])
    for s in range(2, 8):
        household_size_count[s] = int(df.loc[df['NAME'] == location]['B11016_00' + '%i' % (s+1) + 'E'].values[0]) + int(df.loc[df['NAME'] == location]['B11016_0' + '%i' % (s+9) + 'E'].values[0])
    return household_size_count


def process_census_employment_rates(datadir, location, state_location, country_location, year, acs_period):
    """
    Process American Community Survey data for a given year to get employment rates by age as a fraction.

    Args:
        datadir (str)          : file path to the data directory
        location (str)         : name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        year (int)             : the year for the American Community Survey
        acs_period (int)       : the number of years for the American Community Survey

    Returns:
        A dictionary with the employment rates by age as a fraction.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'employment')
    file_path = ls(os.path.join(file_path, 'ACSST{acs_period}Y{year}.S2301_data_with_overlays_{location}*'.format(acs_period=acs_period, year=year, location=location)))[0]

    df = pd.read_csv(file_path)
    columns = {i: 'S2301_C03_00' + '%i' % i + 'E' for i in range(2, 10)}
    for i in range(10, 12):
        columns[i] = 'S2301_C03_0' + '%i' % i + 'E'
    column_age_ranges = {}
    column_age_ranges[2] = np.arange(16, 20)
    column_age_ranges[3] = np.arange(20, 25)
    column_age_ranges[4] = np.arange(25, 30)
    column_age_ranges[5] = np.arange(30, 35)
    column_age_ranges[6] = np.arange(35, 45)
    column_age_ranges[7] = np.arange(45, 55)
    column_age_ranges[8] = np.arange(55, 60)
    column_age_ranges[9] = np.arange(60, 65)
    column_age_ranges[10] = np.arange(65, 75)
    column_age_ranges[11] = np.arange(75, 101)

    employment_rates = dict.fromkeys(np.arange(16, 101), 0)
    for i in column_age_ranges:
        for a in column_age_ranges[i]:
            employment_rates[a] = float(df.loc[df['NAME'] == location][columns[i]].values[0])/100.
    return employment_rates


def process_census_enrollment_rates(datadir, location, state_location, country_location, year, acs_period):
    """
    Process American Community Survey data for a given year to get enrollment rates by age as a fraction.

    Args:
        datadir (str)          : file path to the data directory
        location (str)         : name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        year (int)             : the year for the American Community Survey
        acs_period (int)       : the number of years for the American Community Survey

    Returns:
        A dictionary with the enrollment rates by age as a fraction.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'enrollment')
    file_path = ls(os.path.join(file_path, 'ACSST{acs_period}Y{year}.S1401_data_with_overlays_{location}*'.format(acs_period=acs_period, year=year, location=location)))[0]

    df = pd.read_csv(file_path)
    columns = {i: 'S1401_C02_0' + '%i' % i + 'E' for i in np.arange(14, 30, 2)}
    column_age_ranges = {}
    column_age_ranges[14] = np.arange(3, 5)
    column_age_ranges[16] = np.arange(5, 10)
    column_age_ranges[18] = np.arange(10, 15)
    column_age_ranges[20] = np.arange(15, 18)
    column_age_ranges[22] = np.arange(18, 20)
    column_age_ranges[24] = np.arange(20, 25)
    column_age_ranges[26] = np.arange(25, 35)
    column_age_ranges[28] = np.arange(35, 51)

    enrollment_rates = dict.fromkeys(np.arange(101), 0)
    for i in column_age_ranges:
        for a in column_age_ranges[i]:
            enrollment_rates[a] = float(df.loc[df['NAME'] == location][columns[i]].values[0])/100.
    return enrollment_rates


def write_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets):
    """
    Write age bracket distribution binned to 18 age brackets.

    Args:
        datadir (str)            : file path to the data directory
        location_alias (str)     : more commonly known name of the location
        state_location (str)     : name of the state the location is in
        country_location (str)   : name of the country the location is in
        age_bracket_count (dict) : dictionary of the age count given by 18 brackets
        age_brackets (dict)      : dictionary of the age range for each bracket

    Returns:
        None.
    """
    age_bracket_distr = sp.norm_dic(age_bracket_count)
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_name = os.path.join(file_path, location_alias + '_age_bracket_distr_18.dat')
    f = open(file_name, 'w')
    f.write('age_bracket,percent\n')
    for b in sorted(age_brackets.keys()):
        s = age_brackets[b][0]
        e = age_brackets[b][-1]
        f.write('%i' % s + '_' + '%i' % e + ',' + '%.16f' % age_bracket_distr[b] + '\n')
    f.close()


def write_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets):
    """
    Write age bracket distribution binned to 16 age brackets.

    Args:
        datadir (str)            : file path to the data directory
        location_alias (str)     : more commonly known name of the location
        state_location (str)     : name of the state the location is in
        country_location (str)   : name of the country the location is in
        age_bracket_count (dict) : dictionary of the age count given by 18 brackets
        age_brackets (dict)      : dictionary of the age range for each bracket

    Returns:
        None.
    """
    age_bracket_distr = sp.norm_dic(age_bracket_count)
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_name = os.path.join(file_path, location_alias + '_age_bracket_distr_16.dat')
    f = open(file_name, 'w')
    f.write('age_bracket,percent\n')
    for b in range(15):
        s = age_brackets[b][0]
        e = age_brackets[b][-1]
        f.write('%i' % s + '_' + '%i' % e + ',' + '%.16f' % age_bracket_distr[b] + '\n')
    f.write('%i' % age_brackets[15][0] + '_' + '%i' % age_brackets[max(age_brackets.keys())][-1] + ',' + '%.16f' % np.sum([age_bracket_distr[b] for b in range(15, len(age_bracket_distr))]) + '\n')
    f.close()


def write_gender_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets):
    """
    Write age bracket by gender distribution.

    Args:
        datadir (str)            : file path to the data directory
        location_alias (str)     : more commonly known name of the location
        state_location (str)     : name of the state the location is in
        country_location (str)   : name of the country the location is in
        age_bracket_distr (dict) : dictionary of the age count by gender given by 18 brackets
        age_brackets (dict)      : dictionary of the age range for each bracket

    Returns:
        None.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_name = os.path.join(file_path, location_alias + '_gender_fraction_by_age_bracket_18.dat')
    f = open(file_name, 'w')
    f.write('age_bracket,fraction_male,fraction_female\n')
    for b in sorted(age_brackets.keys()):
        s = age_brackets[b][0]
        e = age_brackets[b][-1]
        mcount = age_bracket_count_by_gender['male'][b]
        fcount = age_bracket_count_by_gender['female'][b]
        mfrac = float(mcount)/(mcount + fcount)
        ffrac = float(fcount)/(mcount + fcount)
        f.write('%i' % s + '_' + '%i' % e + ',' + '%.16f' % mfrac + ',' + '%.16f' % ffrac + '\n')
    f.close()


def write_gender_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets):
    """
    Write age bracket by gender distribution binned to 16 age brackets.

    Args:
        datadir (str)            : file path to the data directory
        location_alias (str)     : more commonly known name of the location
        state_location (str)     : name of the state the location is in
        country_location (str)   : name of the country the location is in
        age_bracket_distr (dict) : dictionary of the age count by gender given by 18 brackets
        age_brackets (dict)      : dictionary of the age range for each bracket

    Returns:
        None.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age_distributions')
    file_name = os.path.join(file_path, location_alias + '_gender_fraction_by_age_bracket_16.dat')
    f = open(file_name, 'w')
    f.write('age_bracket,fraction_male,fraction_female\n')
    for b in range(15):
        s = age_brackets[b][0]
        e = age_brackets[b][-1]
        mcount = age_bracket_count_by_gender['male'][b]
        fcount = age_bracket_count_by_gender['female'][b]
        mfrac = float(mcount)/(mcount + fcount)
        ffrac = float(fcount)/(mcount + fcount)
        f.write('%i' % s + '_' + '%i' % e + ',' + '%.16f' % mfrac + ',' + '%.16f' % ffrac + '\n')
    s = age_brackets[15][0]
    e = age_brackets[max(age_brackets.keys())][-1]
    mcount = np.sum([age_bracket_count_by_gender['male'][b] for b in range(15, len(age_brackets))])
    fcount = np.sum([age_bracket_count_by_gender['female'][b] for b in range(15, len(age_brackets))])
    mfrac = float(mcount)/(mcount + fcount)
    ffrac = float(fcount)/(mcount + fcount)
    f.write('%i' % s + '_' + '%i' % e + ',' + '%.16f' % mfrac + ',' + '%.16f' % ffrac + '\n')
    f.close()


def read_household_size_count(datadir, location_alias, state_location, country_location):
    """
    Get household size count dictionary.

    Args:
        datadir (str)          : file path to the data directory
        location_alias (str)   : more commonly known name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in

    Returns:
        dict: A dictionary of the household size count.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household_size_distributions')
    file_name = os.path.join(file_path, location_alias + '_household_size_count.dat')
    df = pd.read_csv(file_name, delimiter=',')
    return dict(zip(df.household_size, df.size_count))


def write_household_size_count(datadir, location_alias, state_location, country_location, household_size_count):
    """
    Write household size count.

    Args:
        datadir (str)               : file path to the data directory
        location_alias (str)        : more commonly known name of the location
        state_location (str)        : name of the state the location is in
        country_location (str)      : name of the country the location is in
        household_size_count (dict) : dictionary of the household size count.

    Returns:
        None.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household_size_distributions')
    file_name = os.path.join(file_path, location_alias + '_household_size_count.dat')
    f = open(file_name, 'w')
    f.write('household_size,percent\n')
    for s in sorted(household_size_count.keys()):
        f.write('%i' % s + ',' + '%.16f' % household_size_count[s] + '\n')
    f.close()


def write_household_size_distr(datadir, location_alias, state_location, country_location, household_size_count):
    """
    Write household size distribution.

    Args:
        datadir (str)               : file path to the data directory
        location_alias (str)        : more commonly known name of the location
        state_location (str)        : name of the state the location is in
        country_location (str)      : name of the country the location is in
        household_size_count (dict) : dictionary of the household size count.

    Returns:
        None.
    """
    household_size_distr = sp.norm_dic(household_size_count)
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'household_size_distributions')
    file_name = os.path.join(file_path, location_alias + '_household_size_distr.dat')
    f = open(file_name, 'w')
    f.write('household_size,percent\n')
    for s in sorted(household_size_count.keys()):
        f.write('%i' % s + ',' + '%.16f' % household_size_distr[s] + '\n')
    f.close()


def write_employment_rates(datadir, location_alias, state_location, country_location, employment_rates):
    """
    Write employment rates by age as a fraction.

    Args:
        datadir (str)               : file path to the data directory
        location_alias (str)        : more commonly known name of the location
        state_location (str)        : name of the state the location is in
        country_location (str)      : name of the country the location is in
        household_size_count (dict) : dictionary of the household size count.

    Returns:
        None.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'employment_rates')
    file_name = os.path.join(file_path, location_alias + '_employment_rates_by_age.dat')
    f = open(file_name, 'w')
    f.write('Age,Percent\n')
    for a in sorted(employment_rates.keys()):
        f.write('%i' % a + ',' + '%.3f' % employment_rates[a] + '\n')
    f.close()


def write_enrollment_rates(datadir, location_alias, state_location, country_location, enrollment_rates):
    """
    Write employment rates by age as a fraction.

    Args:
        datadir (str)               : file path to the data directory
        location_alias (str)        : more commonly known name of the location
        state_location (str)        : name of the state the location is in
        country_location (str)      : name of the country the location is in
        household_size_count (dict) : dictionary of the household size count.

    Returns:
        None.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'enrollment_rates')
    file_name = os.path.join(file_path, location_alias + '_enrollment_rates_by_age.dat')
    f = open(file_name, 'w')
    f.write('Age,Percent\n')
    for a in sorted(enrollment_rates.keys()):
        f.write('%i' % a + ',' + '%.3f' % enrollment_rates[a] + '\n')
    f.close()


if __name__ == '__main__':

    datadir = sp.datadir

    # location = 'Portland-Vancouver-Hillsboro-OR-WA-Metro-Area'
    # location_alias = 'portland_metro'
    location = 'Oregon'
    location_alias = 'Oregon'
    state_location = 'Oregon'

    # location = 'Washington'
    # location_alias = 'Washington'
    # state_location = 'Washington'
    country_location = 'usa'

    year = 2019
    acs_period = 1

    # age_bracket_count, age_brackets = process_census_age_counts(datadir, location, state_location, country_location, year, acs_period)
    # write_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets)
    # write_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets)

    # age_bracket_count_by_gender, age_brackets = process_census_age_counts_by_gender(datadir, location, state_location, country_location, year, acs_period)
    # write_gender_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets)
    # write_gender_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets)

    # household_size_count = read_household_size_count(datadir, location, state_location, country_location)
    # household_size_count = process_census_household_size_count(datadir, location, state_location, country_location, year, acs_period)
    # write_household_size_distr(datadir, location_alias, state_location, country_location, household_size_count)
    # write_household_size_count(datadir, location_alias, state_location, country_location, household_size_count)

    # employment_rates = process_census_employment_rates(datadir, location, state_location, country_location, year, acs_period)
    # write_employment_rates(datadir, location_alias, state_location, country_location, employment_rates)

    # enrollment_rates = process_census_enrollment_rates(datadir, location, state_location, country_location, year, acs_period)
    # write_enrollment_rates(datadir, location_alias, state_location, country_location, enrollment_rates)
