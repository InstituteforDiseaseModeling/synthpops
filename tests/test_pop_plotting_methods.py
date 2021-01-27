"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import synthpops.plotting as sppl
import numpy as np
import matplotlib as mplt
import pytest


# parameters to generate a test population
pars = dict(
    n                               = 10e3,
    rand_seed                       = 123,
    max_contacts                    = None,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    household_method                = 'fixed_ages',
    smooth_ages                     = True,
    window_length                   = 7,  # window for averaging the age distribution

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
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
pars = sc.objdict(pars)


@pytest.mark.parametrize("pars", [pars])
def test_plot_age_distribution_comparison(pars):
    """
    Test that the age comparison plotting method in sp.Pop class works.
    """
    sp.logger.info("Test that the age comparison plotting method in sp.Pop class works.")

    pop = sp.Pop(**pars)

    pop_age_count = dict.fromkeys(np.arange(101), 0)
    for i, person in pop.popdict.items():
        pop_age_count[person['age']] += 1

    # print(pop_age_count)
    print('from pop')
    print(sorted(pop_age_count.items(), key=lambda x: x[1], reverse=True))
    print()
    popdict = pop.to_dict()

    popdict_age_count = dict.fromkeys(np.arange(101), 0)
    for i, person in popdict.items():
        popdict_age_count[person['age']] += 1

    # print(popdict_age_count)
    print('from popdict')
    print(sorted(popdict_age_count.items(), key=lambda x: x[1], reverse=True))
    print()

    kwargs = sc.dcp(pars)
    kwargs['figname'] = 'new_name'

    fig, ax = pop.plot_age_distribution_comparison(**kwargs)
    return fig, ax, popdict


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    fig0, ax0, sppopdict = test_plot_age_distribution_comparison(pars)

    import covasim as cv
    from collections import Counter

    test_pars = sc.dcp(pars)
    test_pars['n'] = 10e3
    fig2, ax2, sppopdict2 = test_plot_age_distribution_comparison(test_pars)

    popdict = cv.make_synthpop(population=sc.dcp(sppopdict2), community_contacts=10)
    print(type(popdict))

    cvpopdict_age_count = Counter(popdict['age'])

    print('from covasim popdict')
    print(sorted(cvpopdict_age_count.items(), key=lambda x: x[1], reverse=True))

    pop_size = test_pars['n']

    school_ids = [None] * int(pop_size)
    teacher_flag = [False] * int(pop_size)
    staff_flag = [False] * int(pop_size)
    student_flag = [False] * int(pop_size)
    school_types = {'pk': [], 'es': [], 'ms': [], 'hs': [], 'uv': []}
    school_type_by_person = [None] * int(pop_size)
    schools = dict()

    for uid, person in sppopdict2.items():
        if person['scid'] is not None:
            school_ids[uid] = person['scid']
            school_type_by_person[uid] = person['sc_type']
            if person['scid'] not in school_types[person['sc_type']]:
                school_types[person['sc_type']].append(person['scid'])
            if person['scid'] in schools:
                schools[person['scid']].append(uid)
            else:
                schools[person['scid']] = [uid]
            if person['sc_teacher'] is not None:
                teacher_flag[uid] = True
            elif person['sc_student'] is not None:
                student_flag[uid] = True
            elif person['sc_staff'] is not None:
                staff_flag[uid] = True

    popdict['school_id'] = np.array(school_ids)
    popdict['schools'] = schools
    popdict['teacher_flag'] = teacher_flag
    popdict['student_flag'] = student_flag
    popdict['staff_flag'] = staff_flag
    popdict['school_types'] = school_types
    popdict['school_type_by_person'] = school_type_by_person

    assert sum(popdict['teacher_flag']), 'Uh-oh, no teachers were found: as a school analysis this is treated as an error'
    assert sum(popdict['student_flag']), 'Uh-oh, no students were found: as a school analysis this is treated as an error'

    # Actually create the people
    people_pars = dict(
        pop_size = pars.n,
        beta_layer = {k:1.0 for k in 'hswcl'}, # Since this is used to define hat layers exist
        beta = 1.0, # TODO: this is required for plotting (people.plot()), but shouldn't be
    )
    people = cv.People(people_pars, strict=False, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'],
                          contacts=popdict['contacts'], school_id=popdict['school_id'],
                          schools=popdict['schools'], school_types=popdict['school_types'],
                          student_flag=popdict['student_flag'], teacher_flag=popdict['teacher_flag'],
                          staff_flag=popdict['staff_flag'], school_type_by_person=popdict['school_type_by_person'])

    kwargs = sc.objdict(sc.dcp(test_pars))
    kwargs.datadir = sp.datadir

    people_age_count = Counter(people.age)
    print('from people')
    print(sorted(people_age_count.items(), key=lambda x: x[1], reverse=True))

    # check that covasim object plotting works
    sppl.plot_age_distribution_comparison(people, **kwargs)

    # check that synthpops dictionary style  object plotting works
    fig1, ax1 = sp.plot_age_distribution_comparison(sp.Pop(**pars).to_dict(), **sc.mergedicts(kwargs, {'color_2': 'indigo', 'color_1': '#ea6075'}))

    # plt.show()


