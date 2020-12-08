'''
Simple benchmarking of individual functions.

To create the test file, set regenerate = True. To test if it matches the saved
version, set regenerate = False.
'''

import sciris as sc
import synthpops as sp

regenerate = True
outfile = 'basic_api.pop'


def test_default():
    ''' Simplest possible usage '''
    pop = sp.Pop(n=1000)
    return pop


def test_example_oop():
    ''' Example SynthPops test of setting multiple parameters '''
    sp.logger.info('Testing basic API')

    pars = dict(
        n                               = 20002,
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

    pop = sp.Pop(**pars)
    popdict = pop.to_dict()

    if not regenerate:
        print('Checking...')
        pop2 = sc.loadobj(outfile)
        assert popdict == pop2, 'Check failed'
        print('Check passed')
    else:
        print('Regenerating regression file...')
        sc.saveobj(outfile, popdict)

    return pop


def test_alternatives():
    ''' Alternative OOP test with different options to test different code paths '''
    sp.logger.info('Testing alternative API')

    n = 2000

    ltcf_pars = sc.objdict(
       with_facilities = True,
       ltcf_staff_age_min = 20,
       ltcf_staff_age_max = 65,
    )

    school_pars = sc.objdict(
       school_mixing_type = 'random',
       average_class_size = 20,
       inter_grade_mixing = True,
       average_student_teacher_ratio = 20,
       teacher_age_min = 22,
       teacher_age_max = 65,
    )

    pops = sc.objdict()

    pops.random_schools = sp.Pop(n=n, ltcf_pars=ltcf_pars, school_pars=school_pars)

    # Generate another population with different parameters
    ltcf_pars2 = sc.mergedicts(ltcf_pars, dict(
        with_facilities=False,
        ))
    school_pars2 = sc.mergedicts(school_pars, dict(
        with_school_types = True,
        school_mixing_type = 'age_clustered',
        ))
    pops.no_ltcf = sp.Pop(n=n, ltcf_pars=ltcf_pars2, school_pars=school_pars2)

    return pops


def test_api(do_plot=False):
    ''' More examples of basic API usage '''
    pop = sp.Pop(n=2000) # default parameters, 5k people
    pop.save('test_api.pop') # save as pickle
    pop.to_json('test_api.json') # save as JSON
    popdict = pop.to_dict() # export from class to standard python object; current default synthpops output
    if do_plot:
        # pop.plot() # do the most obvious plotting thing, whatever that may be...
        pop.plot_people() # equivalent to cv.Sim.people.plot()
        pop.plot_contacts() # equivalent to sp.plot_contact_matrix(popdict)
    return popdict


if __name__ == '__main__':

    T = sc.tic()

    default_pop = test_default()
    example_pop = test_example_oop()
    alt_pops    = test_alternatives()
    api_popdict = test_api(do_plot=True)

    sc.toc(T)
    print('Done.')