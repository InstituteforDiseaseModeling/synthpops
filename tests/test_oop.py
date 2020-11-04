'''
Simple benchmarking of individual functions.

To create the test file, set regenerate = True. To test if it matches the saved
version, set regenerate = False.
'''

import sciris as sc
import synthpops as sp

regenerate = False
outfile = 'basic_api.pop'

pars = dict(
        n                               = 20001,
        rand_seed                       = 123,
        max_contacts                    = None,

        with_industry_code              = 0,
        with_facilities                 = 1,
        with_non_teaching_staff         = 1, # NB: has no effect
        use_two_group_reduction         = 1,
        with_school_types               = 1,

        average_LTCF_degree             = 20,
        ltcf_staff_age_min              = 20,
        ltcf_staff_age_max              = 60,

        school_mixing_type              = 'age_and_class_clustered',
        average_class_size              = 20,
        inter_grade_mixing              = 0.1, # NB: has no effect
        teacher_age_min                 = 25,
        teacher_age_max                 = 75,
        staff_age_min                   = 20,
        staff_age_max                   = 75,

        average_student_teacher_ratio   = 20,
        average_teacher_teacher_degree  = 3,
        average_student_all_staff_ratio = 15,
        average_additional_staff_degree = 20,
)


def test_basic_oop():
    ''' Basic SynthPops test '''
    sp.logger.info('Testing basic API')

    pop = sp.Pop(**pars)
    popdict = pop.to_dict()

    print('Checking...')
    pop2 = sc.loadobj(outfile)
    assert popdict == pop2, 'Check failed'
    print('Check passed')

    return pop


def test_alternative_oop():
    ''' Alternative SynthPops test '''
    sp.logger.info('Testing alternative API')

    pars2 = sc.dcp(pars)
    pars2.update(dict(
        with_school_types = False,
        school_mixing_type = 'random',
    ))

    pop = sp.Pop(**pars2) # Don't check that it's the same as anything, just that it works

    return pop


if __name__ == '__main__':
    T = sc.tic()
    pop = test_basic_oop()
    pop2 = test_alternative_oop()
    sc.toc(T)
    print('Done.')