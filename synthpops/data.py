import json
from jsonobject import *


class SchoolSizeDistributionByType(JsonObject):
    school_type = StringProperty()
    size_distribution = ListProperty(FloatProperty)


class SchoolTypeByAge(JsonObject):
    school_type = StringProperty()
    age_range = ListProperty(FloatProperty)


class Location(JsonObject):
    data_provenance_notices = ListProperty(StringProperty)
    reference_links = ListProperty(StringProperty)
    citations = ListProperty(StringProperty)
    population_age_distribution = ListProperty(ListProperty(FloatProperty))
    employment_rates_by_age = ListProperty(ListProperty(FloatProperty))
    enrollment_rates_by_age = ListProperty(ListProperty(FloatProperty))
    household_head_age_brackets = ListProperty(ListProperty(FloatProperty))
    household_head_age_distribution_by_family_size = ListProperty(ListProperty(FloatProperty))
    household_size_distribution = ListProperty(ListProperty(FloatProperty))
    ltcf_resident_to_staff_ratio_distribution = ListProperty(ListProperty(FloatProperty))
    ltcf_num_residents_distribution = ListProperty(ListProperty(FloatProperty))
    ltcf_num_staff_distribution = ListProperty(ListProperty(FloatProperty))
    school_size_brackets = ListProperty(ListProperty(FloatProperty))
    school_size_distribution = ListProperty(FloatProperty)
    school_size_distribution_by_type = ListProperty(SchoolSizeDistributionByType)
    school_types_by_age = ListProperty(SchoolTypeByAge)
    workplace_size_counts_by_num_personnel = ListProperty(ListProperty(FloatProperty))


def load_location_from_json(json_obj):
    location = Location(json_obj)
    check_location_constraints_satisfied(location)
    return location


def load_location_from_json_str(json_str):
    json_obj = json.loads(json_str)
    return load_location_from_json(json_obj)


def load_location_from_filepath(filepath):
    f = open(filepath, 'r')
    json_obj = json.load(f)
    return load_location_from_json(json_obj)


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

    # TODO complete
    #num_household_age_brackets = len(location.household_head_age_brackets)
    #if location.household_head_age_distribution_by_family_size is not None:
    #    for household_head_age_distribution in location.household_head_age_distribution_by_family_size:

    return [True, None]
