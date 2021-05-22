"""Template for how to create an example data json. You should fill in the TODO
sections and refer to 03_create_data_json_malawi_example.py for more details.
"""
import synthpops as sp
import pandas as pd
import os


def location_data_construct(func):
    """
    Leave the contents of this function as is (do not modify). To fill the
    properties of the location_data, implement the get methods below with
    function name = "get_{property}" appropriate to your own data files. Follow
    this template for any other data fields or properties you wish to fill.

    Args:
        func:

    Returns:

    """
    def add_location_data(*args, **kwargs):
        property = func(*args, **kwargs)
        if property.__class__.__name__ == 'PopulationAgeDistribution':
            getattr(args[0], func.__name__[4:]).append(property)
        else:
            if func.__name__[4:] in ('reference_links', 'citations', 'notes') and property.__class__.__name__ != 'str':
                # append lists of strings if the property is not a single string
                for i in property:
                    getattr(args[0], func.__name__[4:]).append(i)
            else:
                setattr(args[0], func.__name__[4:], property)
        return args[0]
    return add_location_data


@location_data_construct
def get_population_age_distributions(location_data, num_agebrackets):
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
    # TODO: implement
    return


@location_data_construct
def get_employment_rates_by_age(location_data):
    """
    Read in employment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (101, 2) with data
        on the employment rates for ages 0 through 100.
    """
    # TODO: implement
    return


@location_data_construct
def get_enrollment_rates_by_age(location_data):
    """
    Read in enrollment rates from csv files and format the data to add to the
    location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (101, 2) with data
        on the enrollment rates for ages 0 through 100.
    """
    # TODO: implement
    return


@location_data_construct
def get_household_size_distribution(location_data):
    """
    Read in household size distribution from csv files and format the data to
    add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of household sizes, 2) with data
        on the household size distribution.
    """
    # TODO: implement
    return


@location_data_construct
def get_household_head_age_brackets(location_data):
    """
    Read in household head age brackets from csv files and format the data to
    add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of head of age brackets, 2)
        with data on the household size distribution.
    """
    # TODO: implement
    return


@location_data_construct
def get_household_head_age_distribution_by_family_size(location_data):
    """
    Read in head of household age distribution by household size from csv files and format the data to
    add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of household sizes, number of household head of age brackets)
        with data on the household size distribution.
    """
    # TODO: implement
    return


@location_data_construct
def get_workplace_size_counts_by_num_personnel(location_data):
    """
    Read in workplace size distribution data from csv files and format the data
    to add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        array : An array with dimensions (number of workplace size brackets, 3) with data
        on the workplace size distribution.
    """
    # TODO: implement
    return


@location_data_construct
def get_school_types_by_age(location_data):
    """
    Read in the school type and age range data from csv files and format the
    data to add to the location_data json object.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        dict : An dictionary mapping school type to the distinct age range for
        each school type.
    """
    # TODO: implement
    return


@location_data_construct
def get_reference_links(location_data):
    """
    Read in list of reference links for sources.

    Args:
        location_data (sp.Location) : json-based data object for the location

    Returns:
        list : A list of the reference links for the original data.
    """
    # TODO: implement
    return


if __name__ == '__main__':

    # TODO: Change to your location name
    location_name = ''

    # Check if location_name /data_path is set
    if location_name == '':
        raise ValueError("Please specify location name!")

    # Default path to the new json we want to create for your location
    json_filepath = os.path.join(sp.settings.datadir, f'{location_name}.json')
    location_data = sp.Location()

    # add the country as the location_name
    location_data.location_name = location_name

    # TODO: change available age distributions
    # for exampleL age distribution for 96 age brackets and then age distribution mapped to
    # 16 age brackets to match the default contact matrices used in SynthPops
    available_num_agebrackets = []  # e.g.[16, 96]
    for num_agebrackets in available_num_agebrackets:
        # get the PopulationAgeDistribution for a specific number of age brackets
        get_population_age_distributions(location_data, num_agebrackets=num_agebrackets)

    # get the employment rates array and populate the employment rates field
    get_employment_rates_by_age(location_data)

    # get the enrollment rates array and populate the enrollment rates field
    get_enrollment_rates_by_age(location_data)

    # get the household size distribution array and populate the household size distribution field
    get_household_size_distribution(location_data)

    # get the head of household age brackets and populate the household head age brackets field
    get_household_head_age_brackets(location_data)

    # get the head of household age distribution by household or family size and populate the household head age distribution by family size field
    get_household_head_age_distribution_by_family_size(location_data)

    # get the workplace size distribution array and populate the workplace size distribution field
    get_workplace_size_counts_by_num_personnel(location_data)

    # get age ranges by school type and populate the school types by age --- defines an age range for each school type
    get_school_types_by_age(location_data)

    # get reference links and populate the reference links
    # you might do this step repeatedly as you find more data and populate your json
    get_reference_links(location_data)

    # TODO: adding notes to your json --- sometimes you may want to add additional information on how data were inferred
    # you can also implement a method get_note with decorator '@location_data_construct' to return list of notes
    note = ""
    # add note
    location_data.notes.append(note)

    # save the loaded json file
    sp.save_location_to_filepath(location_data, json_filepath)
    print('saved.')

    # check that you can reload the newly created json
    new_location_data = sp.load_location_from_filepath(json_filepath)
    print(f'{new_location_data.location_name} loaded.')
