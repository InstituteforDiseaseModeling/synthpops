"""Create zimbabwe json."""
import synthpops as sp
import pandas as pd
import numpy as np
import os


def get_age_dist_arr(location_data, num_agebrackets=16):
    """
    Read in age distribution data from csv files and format the data to add to
    the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location
        num_agebrackets (int) : the number of age brackets or bins

    Returns:
        array : An array with dimensions (number of age brackets, 3) with data
        on the age distribution.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f"{location_data.location_name}_ages_{num_agebrackets}.csv"))
    age_dist_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age_min', 'age_max'])

    age_dist = sp.PopulationAgeDistribution()
    age_dist.num_bins = len(age_dist_arr)
    age_dist.distribution = age_dist_arr

    return age_dist


def process_employment_rates(location_data):
    """Process and return employment data."""

    raw_data_path = os.path.join(sp.settings.datadir, location_data.location_name)
    em_df = pd.read_csv(os.path.join(raw_data_path, f'{location_data.location_name}_employment_rates_binned_by_age.csv'))
    binned_rates = em_df['percent'].values
    employment_rates = dict.fromkeys(np.arange(101), 0)
    for bi in range(len(em_df)):
        b0 = em_df['age_min'].values[bi]
        b1 = em_df['age_max'].values[bi]

        for a in np.arange(b0, b1 + 1):
            employment_rates[a] = binned_rates[bi]

    employment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(employment_rates)), percent=[employment_rates[a] for a in sorted(employment_rates.keys())]))
    return sp.convert_df_to_json_array(employment_rates_df, cols=employment_rates_df.columns, int_cols=['age'])


def process_enrollment_rates(location_data):
    """Process and return enrollment data."""

    raw_data_path = os.path.join(sp.settings.datadir, location_data.location_name)
    en_df = pd.read_csv(os.path.join(raw_data_path, f'{location_data.location_name}_enrollment_rates_binned_by_age.csv'))
    binned_rates = en_df['percent'].values
    enrollment_rates = dict.fromkeys(np.arange(101), 0)
    for bi in range(len(en_df)):
        b0 = en_df['age_min'].values[bi]
        b1 = en_df['age_max'].values[bi]

        for a in range(b0, b1 + 1):
            enrollment_rates[a] = binned_rates[bi]

    enrollment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(enrollment_rates)), percent=[enrollment_rates[a] for a in sorted(enrollment_rates.keys())]))
    return sp.convert_df_to_json_array(enrollment_rates_df, cols=enrollment_rates_df.columns, int_cols=['age'])


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
    return household_size_dist_arr


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

    return school_type_age_ranges


def get_reference_links(location_data):
    """
    Read in list of reference links for sources.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        list : A list of the reference links for the original data.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, 'sources.txt'))
    return df.reference_links.values


if __name__ == '__main__':

    location_name = "Zimbabwe"

    # path to national data for Zimbabwe in separate files
    data_path = os.path.join(sp.settings.datadir, location_name)

    # path to the new json we want to create for Zimbabwe
    json_filepath = os.path.join(sp.settings.datadir, f'{location_name}.json')

    # check if the file already exists and if not, create one
    try:
        location_data = sp.load_location_from_filepath(json_filepath)
    except:
        location_data = sp.Location()

    # add the country as the location_name
    location_data.location_name = location_name

    # available age distributions for Malawi recently (from 2018)
    # age distribution for 19 age brackets and then age distribution mapped to
    # 16 age brackets to match the default contact matrices used in SynthPops

    for num_agebrackets in [16, 19]:
        # get the PopulationAgeDistribution for a specific number of age brackets
        age_dist = get_age_dist_arr(location_data, num_agebrackets=num_agebrackets)
        # populate the data field
        location_data.population_age_distributions.append(age_dist)

    # get the employment rates array
    employment_rates_arr = process_employment_rates(location_data)
    # populate the employment rates field
    location_data.employment_rates_by_age = employment_rates_arr

    # get the enrollment rates array
    enrollment_rates_arr = process_enrollment_rates(location_data)
    # populate the enrollment rates field
    location_data.enrollment_rates_by_age = enrollment_rates_arr

    # get the household size distribution array
    household_size_dist_arr = get_household_size_dist_arr(location_data)
    # populate the household size distribution field
    location_data.household_size_distribution = household_size_dist_arr

    # get age ranges by school type
    school_types_by_age = get_school_type_age_ranges(location_data)
    # populate the school types by age --- defines an age range for each school type
    location_data.school_types_by_age = school_types_by_age

    # get reference links
    reference_links = get_reference_links(location_data)

    # populate the reference links --- you might do this step repeatedly as you find more data and populate your json
    for link in reference_links:
        location_data.reference_links.append(link)

    # save the loaded json file
    sp.save_location_to_filepath(location_data, json_filepath)

    # check that you can reload the newly created json
    new_location_data = sp.load_location_from_filepath(os.path.join(sp.settings.datadir, f"{location_name}.json"))
