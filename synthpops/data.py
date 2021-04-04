import json
import jsbeautifier
from jsonobject import *
from jsonobject.base_properties import DefaultProperty
from jsonobject.containers import JsonDict
import os
from . import config as cfg
from . import logger


class SchoolSizeDistributionByType(JsonObject):
    """Class for the school size distribution by school type."""
    school_type = StringProperty()
    # length should be len(location.school_size_distribution)
    size_distribution = ListProperty(DefaultProperty)


class SchoolTypeByAge(JsonObject):
    """Class for the school type by age range."""
    school_type = StringProperty()
    # [min_age, max_age]
    age_range = ListProperty(DefaultProperty)


class Location(JsonObject):
    """
    Class for the data object of the location.

    The general use case of this is to use a filepath, and the parent data is
    parsed from the filepath. DefaultProperty type handles either a scalar or
    json object. We allow a json object mainly for testing of inheriting from
    a parent specified directly in the json.

    Most users will want to populate this with a relative or absolute file path.

    Notes:
        The structures for the population age distribution need to be updated
        to be flexible to take in a parameter for the number of age brackets
        to generate the population age distribution structure.
    """
    location_name = StringProperty()
    data_provenance_notices = ListProperty(StringProperty)
    reference_links = ListProperty(StringProperty)
    citations = ListProperty(StringProperty)
    notes = ListProperty(StringProperty)

    parent = DefaultProperty()

    population_age_distribution_16 = ListProperty(
        # [min_age, max_age, percentage]
        ListProperty(DefaultProperty)
    )

    population_age_distribution_18 = ListProperty(
        # [min_age, max_age, percentage]
        ListProperty(DefaultProperty)
    )

    population_age_distribution_20 = ListProperty(
        # [min_age, max_age, percentage]
        ListProperty(DefaultProperty)
    )

    employment_rates_by_age = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    enrollment_rates_by_age = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    household_head_age_brackets = ListProperty(
        # [age_min, age_max]
        ListProperty(DefaultProperty)
    )

    household_head_age_distribution_by_family_size = ListProperty(
        # length should be len(household_head_age_brackets) + 1
        # The first entry is the family size, the rest of the entries fill in the household head age counts for
        # each household head age bracket.
        # [family_size, count_1, count_2, ...]
        ListProperty(DefaultProperty)
    )

    household_size_distribution = ListProperty(
        # [size, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_resident_to_staff_ratio_distribution = ListProperty(
        # [ratio_low, ratio_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_num_residents_distribution = ListProperty(
        # [num_residents_low, num_residents_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_num_staff_distribution = ListProperty(
        # [num_staff_low, num_staff_hi, percentage]
        ListProperty(DefaultProperty)
    )

    ltcf_use_rate_distribution = ListProperty(
        # [age, percentage]
        ListProperty(DefaultProperty)
    )

    school_size_brackets = ListProperty(
        # [school_size_low, school_size_hi]
        ListProperty(DefaultProperty)
    )

    school_size_distribution = ListProperty(DefaultProperty)

    # The length of size_distribution needs to equal the length of school_size_brackets
    school_size_distribution_by_type = ListProperty(SchoolSizeDistributionByType)

    school_types_by_age = ListProperty(SchoolTypeByAge)

    workplace_size_counts_by_num_personnel = ListProperty(
        # [num_personnel_low, num_personnel_hi, count]
        ListProperty(DefaultProperty)
    )

    def get_list_properties(self):
        """
        Get the properties of the location data object as a list.

        Returns:
            list: A list of the properties of the location data object.
        """
        return [p for p in self if type(getattr(self, p)) is JsonArray]

    def get_population_age_distribution(self, nbrackets):
        """
        Get the age distribution of the population aggregated to nbrackets
        age brackets. Currently we support 16, 18, and, 20 nbrackets and
        return a container for the age distribution with nbrackets.

        Args:
            nbrackets (int): the number of age brackets the age distribution is aggregated to

        Returns:
            list: A list of the probability age distribution values indexed by
            the bracket number.
        """
        if nbrackets not in [16, 18, 20]:
            raise RuntimeError(f"Unsupported value for nbrackets: {nbrackets}")

        dists = {
            16: self.population_age_distribution_16,
            18: self.population_age_distribution_18,
            20: self.population_age_distribution_20,
        }

        dist = dists[nbrackets]
        return dist


def populate_parent_data_from_file_path(location, parent_file_path):
    """
    Loading a location json data object with necessary data fields filled from
    the parent location using the parent location file path.

    Args:
        location (json)        : json data object for the location  # parameter name should probably change to reflect that better
        parent_file_path (str) : file path to the parent location

    Returns:
        json: The location json data object with necessary data fields filled
        from the parent location.
    """
    logger.debug(f"Loading parent location from filepath [{parent_file_path}]")
    try:
        parent_obj = load_location_from_filepath(parent_file_path)
        location = populate_parent_data_from_json_obj(location, parent_obj)
    except:
        logger.warn(f"You may have an invalid data configuration: couldn't load parent "
                    f"from filepath [{parent_file_path}] for location [{location.location_name}]")
    return location


def populate_parent_data_from_json_obj(location, parent):
    """
    Loading a location data object with necessary data fields filled from
    the parent location json.

    Args:
        location (json): json data object for the location  # parameter name should probably change to reflect that better
        parent (json): json data object for the parent location  # parameter name should probably change to reflect that better

    Returns:
        json: The location json data object with necessary data fields filled
        from the parent location.
    """
    if parent.parent is not None:
        populate_parent_data(parent)

    for list_property in location.get_list_properties():
        child_value = getattr(location, list_property)
        if len(child_value) == 0 and str(list_property) in parent:
            parent_value = parent[str(list_property)]
            if len(parent_value) > 0:
                setattr(location, list_property, parent_value)

    return location


def populate_parent_data(location):
    if location.parent is None:
        return location

    parent = location.parent

    if type(parent) is str:
        if len(parent) == 0:
            return location
        return populate_parent_data_from_file_path(location, parent)

    if type(parent) is JsonDict:
        parent_location = Location(parent)
        return populate_parent_data_from_json_obj(location, parent_location)

    raise RuntimeError(f'Invalid type for parent field: [{type(parent)}]')


def load_location_from_json(json_obj):
    location = Location(json_obj)
    check_location_constraints_satisfied(location)
    populate_parent_data(location)
    return location


def load_location_from_json_str(json_str):
    json_obj = json.loads(json_str)
    return load_location_from_json(json_obj)


def get_relative_path(datadir):
    base_dir = datadir
    if len(cfg.rel_path) > 1:
        base_dir = os.path.join(datadir, *cfg.rel_path)
    return base_dir


def load_location_from_filepath(rel_filepath):
    """
    Loads location from provided relative filepath; relative to cfg.datadir.

    Args:
        rel_filepath:

    Returns:

    """
    filepath = os.path.join(get_relative_path(cfg.datadir), rel_filepath)
    logger.debug(f"Opening location from filepath [{filepath}]")
    f = open(filepath, 'r')
    json_obj = json.load(f)
    return load_location_from_json(json_obj)


def save_location_to_filepath(location, abs_filepath):
    """
    Saves location data to provided absolute filepath.

    Args:
        location:
        abs_filepath:

    Returns:

    """
    logger.debug(f"Saving location to filepath [{abs_filepath}]")
    location_json = location.to_json()

    options = jsbeautifier.default_options()
    options.indent_size = 2
    location_json = jsbeautifier.beautify(json.dumps(location_json), options)

    with open(abs_filepath, 'w') as f:
        f.write(location_json)
        #json.dump(location_json, f, indent=2)


def check_location_constraints_satisfied(location):
    """
    Checks a number of constraints that need to be satisfied, above and
    beyond the schema.

    Args:
        location:

    Returns:
        Nothing

    Raises:
        RuntimeError with a description if one of the constraints is
        not satisfied.

    """
    [status, msg] = are_location_constraints_satisfied(location)
    if not status:
        raise RuntimeError(msg)


def are_location_constraints_satisfied(location):
    """
    Checks a number of constraints that need to be satisfied, above and
    beyond the schema.

    Args:
        location:

    Returns:
        [True, None] If all constraints are satisfied.
        [False, string] If a constraint is violated.

    """

    for f in [check_location_name,
              check_population_age_distribution_16,
              check_employment_rates_by_age,
              check_enrollment_rates_by_age,
              check_household_age_brackets,
              check_household_head_age_distributions_by_family_size,
              check_household_size_distribution,
              check_ltcf_resident_to_staff_ratio_distribution,
              check_ltcf_num_residents_distribution,
              check_ltcf_num_staff_distribution,
              check_school_size_brackets,
              check_school_size_distribution,
              check_school_size_distribution_by_type,
              check_school_types_by_age,
              check_workplace_size_counts_by_num_personnel,
              ]:
        [status, msg] = f(location)
        if not status:
            return [status, msg]

    return [True, None]


def check_array_of_arrays_entry_lens(location, expected_len, property_name):
    arr = getattr(location, property_name)
    for [k, bracket] in enumerate(arr):
        if not len(bracket) == expected_len:
            return [False,
                    f"Entry [{k}] in {property_name} has invalid length: [{len(bracket)}]; should be [{expected_len}]"]
    return [True, None]


def check_location_name(location):
    if location.location_name is not None and len(location.location_name) > 0:
        return [True, ""]

    return [False, "location_name must be specified"]


def check_population_age_distribution_16(location):
    if len(location.population_age_distribution_16) == 0:
        return [True, ""]
    if len(location.population_age_distribution_16) != 16:
        return [False, f"Invalid length for {location.population_age_distribution_16}: "
                       f"{len(location.population_age_distribution_16)}"]
    return check_array_of_arrays_entry_lens(location, 3, 'population_age_distribution_16')


def check_population_age_distribution_18(location):
    if len(location.population_age_distribution_18) == 0:
        return [True, ""]
    if len(location.population_age_distribution_18) != 18:
        return [False, f"Invalid length for {location.population_age_distribution_18}: "
                       f"{len(location.population_age_distribution_18)}"]
    return check_array_of_arrays_entry_lens(location, 3, 'population_age_distribution_18')


def check_population_age_distribution_20(location):
    if len(location.population_age_distribution_20) == 0:
        return [True, ""]
    if len(location.population_age_distribution_20) != 20:
        return [False, f"Invalid length for {location.population_age_distribution_20}: "
                       f"{len(location.population_age_distribution_20)}"]
    return check_array_of_arrays_entry_lens(location, 3, 'population_age_distribution_20')


def check_employment_rates_by_age(location):
    return check_array_of_arrays_entry_lens(location, 2, 'employment_rates_by_age')


def check_enrollment_rates_by_age(location):
    return check_array_of_arrays_entry_lens(location, 2, 'enrollment_rates_by_age')


def check_household_age_brackets(location):
    return check_array_of_arrays_entry_lens(location, 2, 'household_head_age_brackets')


def check_household_head_age_distributions_by_family_size(location):
    num_household_age_brackets = len(location.household_head_age_brackets)

    for [k, household_head_age_distribution] in enumerate(location.household_head_age_distribution_by_family_size):
        expected_len = 1 + num_household_age_brackets
        actual_len = len(household_head_age_distribution)
        if not actual_len == expected_len:
            return [False,
                    f"Entry [{k}] in household_head_age_distribution_by_family_size has invalid length: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_household_size_distribution(location):
    return check_array_of_arrays_entry_lens(location, 2, 'household_size_distribution')


def check_ltcf_resident_to_staff_ratio_distribution(location):
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_resident_to_staff_ratio_distribution')


def check_ltcf_num_residents_distribution(location):
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_num_residents_distribution')


def check_ltcf_num_staff_distribution(location):
    return check_array_of_arrays_entry_lens(location, 3, 'ltcf_num_staff_distribution')


def check_school_size_brackets(location):
    return check_array_of_arrays_entry_lens(location, 2, 'school_size_brackets')


def check_school_size_distribution(location):
    # TODO: decide if there is a check we should apply here.
    return [True, None]


def check_school_size_distribution_by_type(location):
    num_school_size_brackets = len(location.school_size_brackets)

    for [k, bracket] in enumerate(location.school_size_distribution_by_type):
        expected_len = num_school_size_brackets
        actual_len = len(bracket.size_distribution)
        if not actual_len == num_school_size_brackets:
            return [False,
                    f"Entry [{k} - {bracket.school_type}] in school_size_distribution_by_type has invalid length for size_distribution: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_school_types_by_age(location):
    for [k, bracket] in enumerate(location.school_types_by_age):
        expected_len = 2
        actual_len = len(bracket.age_range)
        if not actual_len == expected_len:
            return [False,
                    f"Entry [{k} - {bracket.school_type}] in school_types_by_age has invalid length for age_range: [{actual_len}]; should be [{expected_len}]"]
    return [True, None]


def check_workplace_size_counts_by_num_personnel(location):
    return check_array_of_arrays_entry_lens(location, 3, 'workplace_size_counts_by_num_personnel')
