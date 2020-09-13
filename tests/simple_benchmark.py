'''
Simple benchmarking of individual functions
'''

import sciris as sc
import synthpops as sp

to_profile = 'make_contacts_from_microstructure_objects' # Must be one of the options listed below

func_options = {
'make_population' : sp.make_population, # Basic function
    'generate_microstructure_with_facilities': sp.generate_microstructure_with_facilities, # Main function with facilities
        'custom_generate_all_households': sp.custom_generate_all_households, # 20%
    'generate_synthetic_population': sp.generate_synthetic_population, # Main function without facilities
        'generate_all_households': sp.generate_all_households, # 15%
        'assign_rest_of_workers' : sp.contact_networks.assign_rest_of_workers, # 10%
        'make_contacts_from_microstructure_objects':  sp.make_contacts_from_microstructure_objects, # 60%
    'trim_contacts': sp.trim_contacts, # This is where much of the time goes for loading a population
    'generate_larger_households'          : sp.contact_networks.generate_larger_households,
    'make_popdict'                        : sp.make_popdict,
    'make_contacts'                       : sp.make_contacts,
    'sample_n_contact_ages'               : sp.sample_n_contact_ages,
    'generate_living_alone'               : sp.contact_networks.generate_living_alone,
    'generate_household_head_age_by_size' : sp.contact_networks.generate_household_head_age_by_size,
    'sample_from_range'                   : sp.sampling.sample_from_range,
}

pars = dict(
        n                               = 20000,
        rand_seed                       = 123,
        max_contacts                    = None,
        generate                        = True,

        with_industry_code              = 0,
        with_facilities                 = 0,
        with_non_teaching_staff         = 1,
        use_two_group_reduction         = 1,
        with_school_types               = 1,

        average_LTCF_degree             = 20,
        ltcf_staff_age_min              = 20,
        ltcf_staff_age_max              = 60,

        school_mixing_type              = 'age_and_class_clustered',
        average_class_size              = 20,
        inter_grade_mixing              = 0.1,
        teacher_age_min                 = 25,
        teacher_age_max                 = 75,
        staff_age_min                   = 20,
        staff_age_max                   = 75,

        average_student_teacher_ratio   = 20,
        average_teacher_teacher_degree  = 3,
        average_student_all_staff_ratio = 15,
        average_additional_staff_degree = 20,
)


def run():
    pop = sp.make_population(**pars)
    return pop

sc.profile(run=run, follow=func_options[to_profile])