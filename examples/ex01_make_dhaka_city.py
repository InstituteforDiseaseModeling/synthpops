"""
An example generating a population of Dhaka City, Bangladesh.
"""
import numpy as np
import sciris as sc
import synthpops as sp

pars = sc.objdict(
    n                               = 20e3,
    rand_seed                       = 123,

    country_location                = 'bangladesh',
    state_location                  = 'dhaka',
    location                        = 'dhaka_city',
    sheet_name                      = 'Bangladesh',

    household_method                = 'fixed_ages',
    smooth_ages                     = 1,

    with_facilities                 = 0,
    average_class_size              = 35,
    average_student_teacher_ratio   = 35,
    average_student_all_staff_ratio = 30, # an estimate could turn this off

    with_school_types               = 1,
    school_mixing_type              = dict(
    pk                              = 'age_and_class_clustered', # 5 year olds
    es                              = 'age_and_class_clustered', # 6-10 year olds
    ss                              = 'age_clustered', # 11-15 year olds
    hs                              = 'random', # 16-18 year olds
    uv                              = 'random' # 19-23 year olds
    )
)

dhaka_pop = sp.Pop(**pars)

# print(dhaka_pop.summary)
# print(dhaka_pop.information)
dhaka_pop.summarize()
