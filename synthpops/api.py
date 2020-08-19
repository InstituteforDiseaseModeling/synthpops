"""
This module provides the layer for communicating with the agent-based model Covasim.
"""

import sciris as sc
import synthpops as sp

# Put this here so it's accessible as sp.api.popsize_choices
popsize_choices = [5000,
                   10000,
                   20000,
                   50000,
                   100000,
                   120000,
                ]


def make_population(n=None, max_contacts=None, generate=None, with_industry_code=False, with_facilities=False,
                    use_two_group_reduction=True, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
                    with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
                    average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
                    with_non_teaching_staff=False,
                    average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
                    rand_seed=None):
    '''
    Make a full population network including both people (ages, sexes) and contacts using Seattle, Washington cached data.
    Args:
        n (int)                                 : The number of people to create.
        max_contacts (dict)                     : A dictionary for maximum number of contacts per layer: keys must be "W" (work).
        generate (bool)                         : If True, generate a new population. Else, look for cached population and if those are not available, generate a new population.
        with_industry_code (bool)               : If True, assign industry codes for workplaces, currently only possible for cached files of populations in the US.
        with_facilities (bool)                  : If True, create long term care facilities, currently only available for locations in the US.
        use_two_group_reduction (bool)          : If True, create long term care facilities with reduced contacts across both groups.
        average_LTCF_degree (float)             : default average degree in long term care facilities.
        ltcf_staff_age_min (int)                : Long term care facility staff minimum age.
        ltcf_staff_age_max (int)                : Long term care facility staff maximum age.
        with_school_types (bool)                : If True, creates explicit school types.
        school_mixing_type (str or dict)                : The mixing type for schools, 'random', 'age_clustered', or 'age_and_class_clustered' if string, and a dictionary of these by school type otherwise.
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

    Returns:
        network (dict): A dictionary of the full population with ages and connections.
    '''

    if rand_seed is not None:
        sp.set_seed(rand_seed)

    default_n = 10000
    default_max_contacts = {'W': 20}  # this can be anything but should be based on relevant average number of contacts for the population under study

    if n is None:
        n = default_n
    n = int(n)

    if n not in popsize_choices:
        if generate is False:
            choicestr = ', '.join([str(choice) for choice in popsize_choices])
            errormsg = f'If generate=False, number of people must be one of {choicestr}, not {n}'
            raise ValueError(errormsg)
        else:
            generate = True  # If n not found in popsize_choices and generate was not False, generate a new population.

    # Default to False, unless LTCF are requested
    if generate is None:
        if with_facilities:
            generate = True
        else:
            generate = False

    max_contacts = sc.mergedicts(default_max_contacts, max_contacts)

    country_location = 'usa'
    state_location = 'Washington'
    location = 'seattle_metro'
    sheet_name = 'United States of America'

    options_args = {}
    options_args['use_microstructure'] = True
    options_args['use_industry_code'] = with_industry_code
    options_args['use_long_term_care_facilities'] = with_facilities
    options_args['use_two_group_reduction'] = use_two_group_reduction
    options_args['with_school_types'] = with_school_types
    options_args['with_non_teaching_staff'] = with_non_teaching_staff

    network_distr_args = {}
    network_distr_args['Npop'] = int(n)

    network_distr_args['average_LTCF_degree'] = average_LTCF_degree

    network_distr_args['average_class_size'] = average_class_size
    network_distr_args['average_student_teacher_ratio'] = average_student_teacher_ratio
    network_distr_args['average_teacher_teacher_degree'] = average_teacher_teacher_degree
    network_distr_args['inter_grade_mixing'] = inter_grade_mixing
    network_distr_args['average_student_all_staff_ratio'] = average_student_all_staff_ratio
    network_distr_args['average_additional_staff_degree'] = average_additional_staff_degree
    network_distr_args['school_mixing_type'] = school_mixing_type

    # Heavy lift 1: make the contacts and their connections
    if not generate:
        # must read in from file, will fail if the data has not yet been generated
        population = sp.make_contacts(location=location, state_location=state_location,
                                      country_location=country_location, sheet_name=sheet_name,
                                      options_args=options_args,
                                      network_distr_args=network_distr_args)
    else:
        # make a new network on the fly
        if with_facilities and with_industry_code:
            errormsg = f'Requesting both long term care facilities and industries by code is not supported yet.'
            raise ValueError(errormsg)
        elif with_facilities:
            population = sp.generate_microstructure_with_facilities(sp.datadir, location=location, state_location=state_location, country_location=country_location, n=n, sheet_name=sheet_name,
                                                                    use_two_group_reduction=use_two_group_reduction, average_LTCF_degree=average_LTCF_degree, ltcf_staff_age_min=ltcf_staff_age_min, ltcf_staff_age_max=ltcf_staff_age_max,
                                                                    with_school_types=with_school_types, school_mixing_type=school_mixing_type, average_class_size=average_class_size, inter_grade_mixing=inter_grade_mixing,
                                                                    average_student_teacher_ratio=average_student_teacher_ratio, average_teacher_teacher_degree=average_teacher_teacher_degree, teacher_age_min=teacher_age_min, teacher_age_max=teacher_age_max,
                                                                    average_student_all_staff_ratio=average_student_all_staff_ratio, average_additional_staff_degree=average_additional_staff_degree, staff_age_min=staff_age_min, staff_age_max=staff_age_max,
                                                                    return_popdict=True )
        else:
            population = sp.generate_synthetic_population(n, sp.datadir, location=location, state_location=state_location, country_location=country_location, sheet_name=sheet_name,
                                                          with_school_types=with_school_types, school_mixing_type=school_mixing_type, average_class_size=average_class_size, inter_grade_mixing=inter_grade_mixing,
                                                          average_student_teacher_ratio=average_student_teacher_ratio, average_teacher_teacher_degree=average_teacher_teacher_degree, teacher_age_min=teacher_age_min, teacher_age_max=teacher_age_max,
                                                          average_student_all_staff_ratio=average_student_all_staff_ratio, average_additional_staff_degree=average_additional_staff_degree, staff_age_min=staff_age_min, staff_age_max=staff_age_max,
                                                          return_popdict=True,
                                                          )

    # Semi-heavy-lift 2: trim them to the desired numbers
    population = sp.trim_contacts(population, trimmed_size_dic=max_contacts, use_clusters=False)

    # Change types
    for key, person in population.items():
        for layerkey in population[key]['contacts'].keys():
            population[key]['contacts'][layerkey] = list(population[key]['contacts'][layerkey])
    return population
