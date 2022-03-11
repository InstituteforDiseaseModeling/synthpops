"""Annotated example of how to create the Malawi json."""
import synthpops as sp
import pandas as pd
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


def get_employment_rates_arr(location_data):
    """
    Read in employment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (101, 2) with data
        on the employment rates for ages 0 through 100.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f'{location_data.location_name}_employment_rates_by_age.csv'))
    employment_rates_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age'])
    return employment_rates_arr


def get_enrollment_rates_arr(location_data):
    """
    Read in enrollment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (101, 2) with data
        on the enrollment rates for ages 0 through 100.
    """
    df = pd.read_csv(os.path.join(sp.settings.datadir, location_data.location_name, f'{location_data.location_name}_enrollment_rates_by_age.csv'))
    enrollment_rates_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age'])
    return enrollment_rates_arr


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
    return workplace_size_dist_arr


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

    location_name = 'Malawi'
    # path to national data for Malawi in separate files
    data_path = os.path.join(sp.settings.datadir, location_name)

    # path to the new json we want to create for Malawi
    json_filepath = os.path.join(sp.settings.datadir, f'{location_name}.json')
    location_data = sp.Location()

    # add the country as the location_name
    location_data.location_name = location_name

    # available age distributions for Malawi recently (from 2018)
    # age distribution for 96 age brackets and then age distribution mapped to
    # 16 age brackets to match the default contact matrices used in SynthPops

    for num_agebrackets in [16, 96]:
        # get the PopulationAgeDistribution for a specific number of age brackets
        age_dist = get_age_dist_arr(location_data, num_agebrackets=num_agebrackets)
        # populate the data field
        location_data.population_age_distributions.append(age_dist)

    # get the employment rates array
    employment_rates_arr = get_employment_rates_arr(location_data)
    # populate the employment rates field
    location_data.employment_rates_by_age = employment_rates_arr

    # get the enrollment rates array
    enrollment_rates_arr = get_enrollment_rates_arr(location_data)
    # populate the enrollment rates field
    location_data.enrollment_rates_by_age = enrollment_rates_arr

    # get the household size distribution array
    household_size_dist_arr = get_household_size_dist_arr(location_data)
    # populate the household size distribution field
    location_data.household_size_distribution = household_size_dist_arr

    # get the workplace size distribution array
    workplace_size_dist_arr = get_workplace_size_dist_arr(location_data)
    # populate the workplace size distribution field
    location_data.workplace_size_counts_by_num_personnel = workplace_size_dist_arr

    # get age ranges by school type
    school_types_by_age = get_school_type_age_ranges(location_data)
    # populate the school types by age --- defines an age range for each school type
    location_data.school_types_by_age = school_types_by_age

    # get reference links
    reference_links = get_reference_links(location_data)

    # populate the reference links --- you might do this step repeatedly as you find more data and populate your json
    for link in reference_links:
        location_data.reference_links.append(link)

    # adding notes to your json --- sometimes you may want to add additional information on how data were inferred
    note = "Secondary school age ranges are unclear. Most sources indicate an 8-4-4 year form of education, however if students take longer to graduate from secondary school then 14-17 years old may be too narrow an age range."
    # add note
    location_data.notes.append(note)

    # save the loaded json file
    sp.save_location_to_filepath(location_data, json_filepath)

    # check that you can reload the newly created json
    new_location_data = sp.load_location_from_filepath(json_filepath)
