'''
Simple run of OOP functionality.
'''

import sciris as sc
import synthpops as sp
import settings
import pytest


def test_default():
    ''' Simplest possible usage '''
    pop = sp.Pop(n=settings.pop_sizes.small)
    return pop


def test_alternatives():
    ''' Alternative OOP test with different options to test different code paths '''
    sp.logger.info('Testing alternative API')

    n = 2 * settings.pop_sizes.small

    ltcf_pars = sc.objdict(
        with_facilities    = True,
        ltcf_staff_age_min = 20,
        ltcf_staff_age_max = 65,
    )

    school_pars = sc.objdict(
        school_mixing_type            = 'random',
        average_class_size            = 20,
        inter_grade_mixing            = 0.1,
        average_student_teacher_ratio = 20,
        teacher_age_min               = 22,
        teacher_age_max               = 65,
    )

    pops = sc.objdict()

    pops.random_schools = sp.Pop(n=n, ltcf_pars=ltcf_pars, school_pars=school_pars)

    # Generate another population with different parameters
    ltcf_pars2 = sc.mergedicts(ltcf_pars, dict(
        with_facilities = False,
    ))
    school_pars2 = sc.mergedicts(school_pars, dict(
        with_school_types  = True,
        school_mixing_type = 'age_clustered',
    ))
    pops.no_ltcf = sp.Pop(n=n, ltcf_pars=ltcf_pars2, school_pars=school_pars2)

    return pops


def test_api(do_plot=False):
    ''' More examples of basic API usage '''
    pop = sp.Pop(n=settings.pop_sizes.medium, rand_seed=0)  # default parameters, 8k people

    pop.save('test_api.pop')  # save as pickle
    pop.to_json('test_api.json')  # save as JSON

    pop2 = sc.dcp(pop)
    pop2.load('test_api.pop')

    sc.saveobj('test_not_pop.obj', [])
    with pytest.raises(TypeError):
        pop2.load('test_not_pop.obj')

    popdict = pop.to_dict()  # export from class to standard python object; current default synthpops output
    if do_plot:
        pop.plot_people()  # equivalent to cv.Sim.people.plot()
        pop.plot_contacts()  # equivalent to sp.plot_contact_matrix(popdict)
        pop.plot_contacts(density_or_frequency='frequency', logcolors_flag=False, aggregate_flag=False)  # test other options
    return popdict


if __name__ == '__main__':

    T = sc.tic()

    default_pop = test_default()
    alt_pops    = test_alternatives()
    api_popdict = test_api(do_plot=True)

    sc.toc(T)
    print('Done.')
