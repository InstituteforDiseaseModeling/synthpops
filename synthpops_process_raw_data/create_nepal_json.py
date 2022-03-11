"""Create nepal json."""
import sciris as sc
import synthpops as sp
import numpy as np
import pandas as pd
import os


def process_age_dists(location_data):
    """
    Read in age distribution data from csv files and format the data to add to
    the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        sp.Location : location_data
    """
    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    age_count_df = pd.read_csv(os.path.join(raw_data_path, 'Nepal-2019.csv'))

    age_count = np.array(age_count_df['M']) + np.array(age_count_df['F'])
    age_dist = age_count / age_count.sum()

    age_bin_labels = age_count_df['Age'].values
    data = dict()
    data['age_min'] = []
    data['age_max'] = []
    data['age_dist'] = []
    for bi, bl in enumerate(age_bin_labels):
        try:
            b = bl.split('-')
            b0, b1 = int(b[0]), int(b[1])
        except:
            b = bl.split('+')
            b0, b1 = int(b[0]), int(b[0])

        data['age_min'].append(b0)
        data['age_max'].append(b1)
        data['age_dist'].append(age_dist[bi])

    for k in data:
        data[k] = np.array(data[k])

    df = pd.DataFrame.from_dict(data)
    age_dist_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age_min', 'age_max'])

    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())
    location_data.population_age_distributions[0].num_bins = len(age_dist_arr)
    location_data.population_age_distributions[0].distribution = age_dist_arr

    data_16 = sc.dcp(data)
    data_16['age_min'] = data_16['age_min'][:-5]
    data_16['age_max'] = data_16['age_max'][:-5]
    data_16['age_max'][-1] = 100
    data_16['age_dist'][-6] = data_16['age_dist'][-6:].sum()
    data_16['age_dist'] = data_16['age_dist'][:-5]

    df_16 = pd.DataFrame.from_dict(data_16)
    age_dist_arr_16 = sp.convert_df_to_json_array(df_16, cols=df_16.columns, int_cols=['age_min', 'age_max'])
    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())
    location_data.population_age_distributions[1].num_bins = len(age_dist_arr_16)
    location_data.population_age_distributions[1].distribution = age_dist_arr_16

    return location_data


def process_employment_rates(location_data):
    """
    Read in employment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        sp.Location : location_data
    """
    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    em_df = pd.read_csv(os.path.join(raw_data_path, 'employment_by_age.csv'))
    age_bin_labels = em_df['Age'].values
    binned_rates = em_df['EmploymentRate'].values
    employment_rates = dict.fromkeys(np.arange(101), 0)
    for bi, bl in enumerate(age_bin_labels):
        try:
            b = bl.split('-')
            b0, b1 = int(b[0]), int(b[1])
        except:
            b = bl.split('+')
            b0, b1 = int(b[0]), int(b[0]) + 10

        for a in np.arange(b0, b1 + 1):
            employment_rates[a] = binned_rates[bi]

    employment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(employment_rates)), percent=[employment_rates[a] for a in sorted(employment_rates.keys())]))
    location_data.employment_rates_by_age = sp.convert_df_to_json_array(employment_rates_df, cols=employment_rates_df.columns, int_cols=['age'])

    return location_data


def process_enrollment_rates(location_data):
    """
    Read in enrollment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        sp.Location : location_data
    """
    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    en_df = pd.read_csv(os.path.join(raw_data_path, 'enrollment_by_age.csv'))
    age_bin_labels = en_df['Age'].values
    binned_rates = en_df['EnrollmentRate'].values
    enrollment_rates = dict.fromkeys(np.arange(101), 0)
    for bi, bl in enumerate(age_bin_labels):
        b = bl.split('-')
        b0, b1 = int(b[0]), int(b[1])

        for a in range(b0, b1 + 1):
            enrollment_rates[a] = binned_rates[bi]

    enrollment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(enrollment_rates)), percent=[enrollment_rates[a] for a in sorted(enrollment_rates.keys())]))
    location_data.enrollment_rates_by_age = sp.convert_df_to_json_array(enrollment_rates_df, cols=enrollment_rates_df.columns, int_cols=['age'])

    return location_data


def get_household_size_dist_arr(location_data):
    """
    Read in household size distribution from csv files and format the data to
    add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of household sizes, 2) with data
        on the household size distribution.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f'{location_data.location_name}_household_sizes.csv'))
    household_size_dist_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['household_size'])
    location_data.household_size_distribution = household_size_dist_arr

    return location_data


def get_school_type_age_ranges(location_data):
    """
    Read in the school type and age range data from csv files and format the
    data to add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        dict : An dictionary mapping school type to the distinct age range for
        each school type.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f"{location_data.location_name}_school_type_age_ranges.csv"))
    arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age_min', 'age_max'])

    school_type_age_ranges = []
    for si in range(len(arr)):
        s = sp.SchoolTypeByAge()
        school_type = arr[si][0]
        s.school_type = school_type
        s.age_range = [arr[si][1], arr[si][2]]
        school_type_age_ranges.append(s)
    location_data.school_types_by_age = school_type_age_ranges

    return location_data


def get_workplace_size_dist_arr(location_data):
    """
    Read in workplace size distribution data from csv files and format the data
    to add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of workplace size brackets, 3) with data
        on the workplace size distribution.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f"{location_data.location_name}_workplace_sizes.csv"))
    workplace_size_dist_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['workplace_size_min', 'workplace_size_max'])
    location_data.workplace_size_counts_by_num_personnel = workplace_size_dist_arr
    return location_data


if __name__ == '__main__':

    location_name = "Nepal"

    # path to the new json we want to create for Zimbabwe
    json_filepath = os.path.join(sp.settings.datadir, f'{location_name}.json')

    # check if the file already exists and if not, create one
    try:
        location_data = sp.load_location_from_filepath(json_filepath)
    except:
        location_data = sp.Location()

    # add the country as the location_name
    location_data.location_name = location_name

    # add age distribution data from raw data files
    location_data = process_age_dists(location_data)

    # add employment rates by age data from raw data files
    location_data = process_employment_rates(location_data)

    # add enrollment rates by age data from raw data files
    location_data = process_enrollment_rates(location_data)

    # add household size distribution from raw data files
    location_data = get_household_size_dist_arr(location_data)

    # add school types by age from raw data files
    location_data = get_school_type_age_ranges(location_data)

    # add workplace size data from raw data files
    location_data = get_workplace_size_dist_arr(location_data)

    # save the loaded json file
    sp.save_location_to_filepath(location_data, json_filepath)

    # check that you can reload the newly created json
    new_location_data = sp.load_location_from_filepath(os.path.join(sp.settings.datadir, f"{location_name}.json"))
