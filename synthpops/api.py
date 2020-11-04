"""
This module provides the layer for communicating with the agent-based model Covasim.
"""

import synthpops as sp
from .config import logger as log


def make_population(*args, **kwargs):
    '''
    Make a full population network including both people (ages, sexes) and contacts using Seattle, Washington data.

    Args:
        n (int)                                 : The number of people to create.
        max_contacts (dict)                     : A dictionary for maximum number of contacts per layer: keys must be "W" (work).
        with_industry_code (bool)               : If True, assign industry codes for workplaces, currently only possible for cached files of populations in the US.
        with_facilities (bool)                  : If True, create long term care facilities, currently only available for locations in the US.
        use_two_group_reduction (bool)          : If True, create long term care facilities with reduced contacts across both groups.
        average_LTCF_degree (float)             : default average degree in long term care facilities.
        ltcf_staff_age_min (int)                : Long term care facility staff minimum age.
        ltcf_staff_age_max (int)                : Long term care facility staff maximum age.
        with_school_types (bool)                : If True, creates explicit school types.
        school_mixing_type (str or dict)        : The mixing type for schools, 'random', 'age_clustered', or 'age_and_class_clustered' if string, and a dictionary of these by school type otherwise.
        average_class_size (float)              : The average classroom size.
        inter_grade_mixing (float)              : The average fraction of mixing between grades in the same school for clustered school mixing types.
        average_student_teacher_ratio (float)   : The average number of students per teacher.
        average_teacher_teacher_degree (float)  : The average number of contacts per teacher with other teachers.
        teacher_age_min (int)                   : The minimum age for teachers.
        teacher_age_max (int)                   : The maximum age for teachers.
        with_non_teaching_staff (bool)          : If True, includes non teaching staff.
        average_student_all_staff_ratio (float) : The average number of students per staff members at school (including both teachers and non teachers).
        average_additional_staff_degree (float) : The average number of contacts per additional non teaching staff in schools.
        staff_age_min (int)                     : The minimum age for non teaching staff.
        staff_age_max (int)                     : The maximum age for non teaching staff.
        rand_seed (int)                         : Start point random sequence is generated from.
        location                  : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name                : sheet name where data is located

    Returns:
        network (dict): A dictionary of the full population with ages and connections.
    '''
    log.debug('make_population()')

    # Heavy lift 1: make the contacts and their connections
    log.debug('Generating a new population...')
    pop = sp.Pop(*args, **kwargs)

    population = pop.to_dict()

    log.debug('make_population(): done.')
    return population
