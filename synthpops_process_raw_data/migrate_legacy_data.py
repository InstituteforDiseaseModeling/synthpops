"""Run this file as a script to migrate legacy files. Run python migrate_legacy_data.py --help for options. """
import data_distributions_legacy as data_distributions_legacy
import synthpops.data as data
import synthpops.config as spconfig
from synthpops import logger as logger
import argparse
import os
import logging
import sys
import re

from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default=None, required=False, help='Source data directory. If unspecified, uses whatever synthpops default is.')
parser.add_argument("--country_location", type=str, default=None, required=True, help='Input country.')
parser.add_argument("--state_location", type=str, default=None, required=False, help='Input state.')
parser.add_argument("--location", type=str, default=None, required=False, help='Input location.')
parser.add_argument("--output_folder", type=str, default=".", required=False, help="Output folder. Default is working directory.")
args = parser.parse_args()


def report_processing_error(data_subject, e):
    """
    Logs a standardized warning message for errors raised during migrating pieces of data.

    Args:
        data_subject : The subject data being migrated, will be included in standardized warning message.
        e            : Exception that was raised during the migration of the subject data.

    Returns:
        None.
    """
    logger.warning(f"Data for {data_subject} may be incomplete, due to error: {e}")


def try_migrate(data_subject, f):
    """
    Attempt a migration as implemented by the bound function f, and logs a standardized warning if an exception
    is raised by f.

    Args:
        data_subject: String indicating the subject data that is being migrated, will be included in standardized
        warning message if raised.

        f: Bound function implementing a data migration.

    Returns:
        None.
    """
    try:
        f()
    except:
        report_processing_error(data_subject, sys.exc_info()[1])


def migrate_population_age_brackets(legacy_age_brackets, new_location):
    num_brackets = len(legacy_age_brackets.keys())
    if num_brackets == 0:
        return
    brackets = []
    for bracket_index, age_range in legacy_age_brackets.items():
        # Insert age bracket with 0.0 percentage for now.
        brackets.append([int(min(age_range)), int(max(age_range)), 0.0])
    population_age_distribution = data.PopulationAgeDistribution()
    population_age_distribution.num_bins = num_brackets
    population_age_distribution.distribution = brackets
    new_location.population_age_distributions.append(population_age_distribution)


def migrate_population_age_brackets_16(datadir, country_location,  state_location, location, new_location,):
    legacy_age_brackets = data_distributions_legacy.get_census_age_brackets(datadir,
                                                                            location,
                                                                            state_location,
                                                                            country_location,
                                                                            nbrackets=16
                                                                            )
    migrate_population_age_brackets(legacy_age_brackets, new_location)


def migrate_population_age_brackets_18(datadir, country_location,  state_location, location, new_location,):
    legacy_age_brackets = data_distributions_legacy.get_census_age_brackets(datadir,
                                                                            location,
                                                                            state_location,
                                                                            country_location,
                                                                            nbrackets=18
                                                                            )
    migrate_population_age_brackets(legacy_age_brackets, new_location)


def migrate_population_age_brackets_20(datadir, country_location,  state_location, location, new_location,):
    legacy_age_brackets = data_distributions_legacy.get_census_age_brackets(datadir,
                                                                            location,
                                                                            state_location,
                                                                            country_location,
                                                                            nbrackets=20
                                                                            )
    migrate_population_age_brackets(legacy_age_brackets, new_location)


def migrate_population_age_distribution(legacy_age_distribution, new_location):
    if len(legacy_age_distribution.items()) == 0:
        return
    nbrackets = len(legacy_age_distribution.keys())
    # The brackets need to have been already migrated, that includes setting num_bins.
    matching_new_dists = [d for d in new_location.population_age_distributions if d.num_bins == nbrackets]
    if len(matching_new_dists) == 0:
        raise RuntimeError(f"Missing new population age distribution w/ num_bins = {nbrackets}")
    new_dist = matching_new_dists[0]
    for bracket_index, dist_percentage in legacy_age_distribution.items():
        new_dist.distribution[bracket_index][2] = dist_percentage


def migrate_population_age_distribution_16(datadir, country_location, state_location, location, new_location):
    legacy_age_distribution = data_distributions_legacy.read_age_bracket_distr(datadir,
                                                                               location,
                                                                               state_location,
                                                                               country_location,
                                                                               16)
    migrate_population_age_distribution(legacy_age_distribution, new_location)


def migrate_population_age_distribution_18(datadir, country_location, state_location, location, new_location):
    legacy_age_distribution = data_distributions_legacy.read_age_bracket_distr(datadir,
                                                                               location,
                                                                               state_location,
                                                                               country_location,
                                                                               18)
    migrate_population_age_distribution(legacy_age_distribution, new_location)


def migrate_population_age_distribution_20(datadir, country_location, state_location, location, new_location):
    legacy_age_distribution = data_distributions_legacy.read_age_bracket_distr(datadir,
                                                                               location,
                                                                               state_location,
                                                                               country_location,
                                                                               20)
    migrate_population_age_distribution(legacy_age_distribution, new_location)


def migrate_employment_rates_by_age(datadir, country_location, state_location, location, new_location):
    legacy_employment_rates = data_distributions_legacy.get_employment_rates(datadir,
                                                                             location,
                                                                             state_location,
                                                                             country_location)
    for age, rate in legacy_employment_rates.items():
        new_location.employment_rates_by_age.append([age, rate])


def migrate_enrollment_rates_by_age(datadir, country_location, state_location, location, new_location):
    legacy_enrollment_rates = data_distributions_legacy.get_school_enrollment_rates(datadir,
                                                                                    location,
                                                                                    state_location,
                                                                                    country_location)
    for age, rate in legacy_enrollment_rates.items():
        new_location.enrollment_rates_by_age.append([age, rate])


def migrate_household_head_age_brackets(datadir, country_location, state_location, location, new_location):
    df = data_distributions_legacy.get_household_head_age_by_size_df(datadir,
                                                                     location,
                                                                     state_location,
                                                                     country_location)

    pattern = re.compile('household_head_age_([\d.]+)_([\d.]+)')
    household_head_age_key_matches = [pattern.match(key) for key in df.keys() if pattern.match(key) is not None]
    new_location.household_head_age_brackets = [
                                                [int(match.group(1)), int(match.group(2))]
                                                for match in household_head_age_key_matches
                                               ]


def migrate_household_head_age_distribution_by_family_size(datadir, country_location, state_location, location, new_location):

    legacy_household_head_age_distribution_by_family_size = \
        data_distributions_legacy.get_head_age_by_size_distr(datadir,
                                                             location,
                                                             state_location,
                                                             country_location)
    for [household_size_index, household_head_age_dist] in enumerate(legacy_household_head_age_distribution_by_family_size):
        household_size = household_size_index + 1
        target_entry = list([household_size])
        target_entry.extend(household_head_age_dist)
        new_location.household_head_age_distribution_by_family_size.append(target_entry)


def migrate_household_size_distribution(datadir, country_location, state_location, location, new_location):

    legacy_household_size_distr = data_distributions_legacy.get_household_size_distr(datadir,
                                                                                     location,
                                                                                     state_location,
                                                                                     country_location)
    for [size, percentage] in legacy_household_size_distr.items():
        new_location.household_size_distribution.append([size, percentage])


def migrate_ltcf_num_residents_distribution(datadir, country_location, state_location, location, new_location):

    legacy_distribution = data_distributions_legacy.get_long_term_care_facility_residents_distr(datadir,
                                                                                                location,
                                                                                                state_location,
                                                                                                country_location)

    legacy_brackets = data_distributions_legacy.get_long_term_care_facility_residents_distr_brackets(datadir,
                                                                                                     location,
                                                                                                     state_location,
                                                                                                     country_location)

    if len(legacy_distribution) != len(legacy_brackets):
        raise RuntimeError(f"Mismatched lengths for distribution and brackets for ltcf num residents distribution "
                           f"for country location [{country_location}], state location [{state_location}], "
                           f"location [{location}]")

    for k, bracket_expanded in legacy_brackets.items():
        bracket_min = float(min(bracket_expanded))
        bracket_max = float(max(bracket_expanded))
        percentage = float(legacy_distribution[k])
        target_entry = [bracket_min, bracket_max, percentage]
        new_location.ltcf_num_residents_distribution.append(target_entry)


def migrate_ltcf_resident_to_staff_ratio_distribution(datadir, country_location, state_location, location,
                                                      new_location):

    legacy_distribution = data_distributions_legacy\
        .get_long_term_care_facility_resident_to_staff_ratios_distr(datadir,
                                                                    location,
                                                                    state_location,
                                                                    country_location)

    legacy_brackets = data_distributions_legacy.\
        get_long_term_care_facility_resident_to_staff_ratios_brackets(datadir,
                                                                      location,
                                                                      state_location,
                                                                      country_location)

    if len(legacy_distribution) != len(legacy_brackets):
        raise RuntimeError(f"Mismatched lengths for distribution and brackets for ltcf reisdent to staff ratio "
                           f"distribution for country location [{country_location}], "
                           f"state location [{state_location}], location [{location}]")

    for k, bracket_expanded in legacy_brackets.items():
        bracket_min = float(min(bracket_expanded))
        bracket_max = float(max(bracket_expanded))
        percentage = float(legacy_distribution[k])
        target_entry = [bracket_min, bracket_max, percentage]
        new_location.ltcf_resident_to_staff_ratio_distribution.append(target_entry)


def migrate_ltcf_use_rate_distribution(datadir, country_location, state_location, location, new_location):

    legacy_distribution = data_distributions_legacy.get_long_term_care_facility_use_rates(datadir,
                                                                                          # This method doesn't take a location parameter.
                                                                                          state_location,
                                                                                          country_location)

    for age, percent in legacy_distribution.items():
        target_entry = [int(age), float(percent)]
        new_location.ltcf_use_rate_distribution.append(target_entry)


def migrate_school_size_brackets(datadir, country_location, state_location, location, new_location):

    legacy_brackets = data_distributions_legacy.get_school_size_brackets(datadir,
                                                                         location,
                                                                         state_location,
                                                                         country_location)

    for k, bracket_expanded in legacy_brackets.items():
        bracket_min = int(min(bracket_expanded))
        bracket_max = int(max(bracket_expanded))
        target_entry = [bracket_min, bracket_max]
        new_location.school_size_brackets.append(target_entry)


def migrate_school_size_distribution(datadir, country_location, state_location, location, new_location):

    legacy_distribution = data_distributions_legacy.get_school_size_distr_by_brackets(datadir,
                                                                                      location,
                                                                                      state_location,
                                                                                      country_location)
    for k, percentage in legacy_distribution.items():
        new_location.school_size_distribution.append(percentage)


def migrate_school_size_distribution_by_type(datadir, country_location, state_location, location, new_location):

    legacy_dists_by_type = data_distributions_legacy.get_school_size_distr_by_type(datadir,
                                                                                  location,
                                                                                  state_location,
                                                                                  country_location)
    for dist_type_key, dist_by_type in legacy_dists_by_type.items():
        target_dist_by_type = data.SchoolSizeDistributionByType()
        target_dist_by_type.school_type = dist_type_key
        for dist_entry_key, percentage in dist_by_type.items():
            target_dist_by_type.size_distribution.append(float(percentage))
        new_location.school_size_distribution_by_type.append(target_dist_by_type)


def migrate_school_types_by_age(datadir, country_location, state_location, location, new_location):

    legacy_type_age_ranges = data_distributions_legacy.get_school_type_age_ranges(datadir,
                                                                                  location,
                                                                                  state_location,
                                                                                  country_location)
    for school_type, ages in legacy_type_age_ranges.items():
        target_type_by_ages = data.SchoolTypeByAge()
        target_type_by_ages.school_type = school_type
        age_min = int(min(ages))
        age_max = int(max(ages))
        target_type_by_ages.age_range = [age_min, age_max]
        new_location.school_types_by_age.append(target_type_by_ages)


def migrate_workplace_size_counts_by_num_personnel(datadir, country_location, state_location, location, new_location):

    legacy_brackets = data_distributions_legacy.get_workplace_size_brackets(datadir,
                                                                            location,
                                                                            state_location,
                                                                            country_location)

    legacy_distribution = data_distributions_legacy.get_workplace_size_distr_by_brackets(datadir,
                                                                                         location,
                                                                                         state_location,
                                                                                         country_location)

    if len(legacy_distribution) != len(legacy_brackets):
        raise RuntimeError(f"Mismatched lengths for distribution and brackets for work size counts country location [{country_location}], state location [{state_location}], location [{location}]")

    for k, bracket_expanded in legacy_brackets.items():
        bracket_min = int(min(bracket_expanded))
        bracket_max = int(max(bracket_expanded))
        count = float(legacy_distribution[k])
        target_entry = [bracket_min, bracket_max, count]
        new_location.workplace_size_counts_by_num_personnel.append(target_entry)


def migrate_legacy_data(datadir, country_location, state_location, location, output_folder):

    logger.info("====================================================================")
    logger.info(f"Migrating data for (country_location, state_location, location) = "
                f"({country_location}, {state_location}, {location})")
    logger.info("====================================================================")

    new_location = data.Location()
    parent = ""
    if country_location is None:
        raise RuntimeError(f"country_location must be provided")
    if location is None and state_location is None:
        new_location.location_name = country_location
        parent = ""
    elif location is None:
        new_location.location_name = f"{country_location}-{state_location}"
        parent = country_location
    else:
        new_location.location_name = f"{country_location}-{state_location}-{location}"
        parent = f"{country_location}-{state_location}"

    new_location.parent = f"{parent}.json" if len(parent) > 0 else  ""

    args = [datadir, country_location, state_location, location, new_location]

    # The key is the subject data being migrated; the value is a bound function that performs the migration.
    migration_functions = {
        "population_age_brackets_16":       partial(migrate_population_age_brackets_16, *args),
        "population_age_brackets_18":       partial(migrate_population_age_brackets_18, *args),
        "population_age_brackets_20":       partial(migrate_population_age_brackets_20, *args),
        "population_age_distribution_16":   partial(migrate_population_age_distribution_16, *args),
        "population_age_distribution_18":   partial(migrate_population_age_distribution_18, *args),
        "population_age_distribution_20":   partial(migrate_population_age_distribution_20, *args),
        "employment_rates_by_age":          partial(migrate_employment_rates_by_age, *args),
        "enrollment_rates_by_age":          partial(migrate_enrollment_rates_by_age, *args),
        "household_head_age_brackets":      partial(migrate_household_head_age_brackets, *args),
        "household_head_age_distribution_by_family_size":
                                            partial(migrate_household_head_age_distribution_by_family_size, *args),
        "household_size_distribution":      partial(migrate_household_size_distribution, *args),
        "ltcf_resident_to_staff_ratio_distribution":
                                            partial(migrate_ltcf_resident_to_staff_ratio_distribution, *args),
        "ltcf_num_residents_distribution":  partial(migrate_ltcf_num_residents_distribution, *args),
        "ltcf_use_rate_distribution":       partial(migrate_ltcf_use_rate_distribution, *args),
        "school_size_brackets":             partial(migrate_school_size_brackets, *args),
        "school_size_distribution":         partial(migrate_school_size_distribution, *args),
        "school_size_distribution_by_type": partial(migrate_school_size_distribution_by_type, *args),
        "school_types_by_age":              partial(migrate_school_types_by_age, *args),
        "workplace_size_counts_by_num_personnel":
                                            partial(migrate_workplace_size_counts_by_num_personnel, *args),
    }

    for key, bound_function in migration_functions.items():
        try_migrate(key, bound_function)

    #make sure output_folder exist
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(output_folder, f"{new_location.location_name}.json")
    data.save_location_to_filepath(new_location, output_filepath)

    logger.info("====================================================================")
    logger.info("--------------------------------------------------------------------")
    logger.info("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    logger.info("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  ")
    logger.info("        - - - - - - - - - - - - - - - - - - - - - - - - - - -       ")
    logger.info("              - - - - - - - - - - - - - - - - - - - - -             ")
    logger.info("                    - - - - - - - - - - - - - - - -                 ")
    logger.info("                        - - - - - - - - - - -                       ")
    logger.info("                              - - - - -                             ")
    logger.info("                                  -                                 ")
    logger.info("                                                                    ")

    return


if __name__ == '__main__':

    datadir = args.datadir
    if datadir is None:
        datadir = spconfig.datadir

    migrate_legacy_data(datadir,
                        args.country_location,
                        args.state_location,
                        args.location,
                        args.output_folder)