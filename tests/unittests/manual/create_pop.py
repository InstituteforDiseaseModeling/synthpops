'''
Generate the population. For use with plot_pop.py. The random seed, population
size, and filename are set at the end. Intentionally uses a lot of specific options
to maximize SynthPops code coverage.
'''

import sciris  as sc
import synthpops as sp
sp.set_nbrackets(20) # Essential for getting the age distribution right

def cache_populations(seed, pop_size, popfile, do_save=True):
    ''' Pre-generate the synthpops population '''

    use_two_group_reduction = True
    average_LTCF_degree = 20
    ltcf_staff_age_min = 20
    ltcf_staff_age_max = 60

    with_school_types = True
    average_class_size = 20
    inter_grade_mixing = 0.1
    average_student_teacher_ratio = 20
    average_teacher_teacher_degree = 3
    teacher_age_min = 25
    teacher_age_max = 75

    with_non_teaching_staff = True
    average_student_all_staff_ratio = 11
    average_additional_staff_degree = 20
    staff_age_min = 20
    staff_age_max = 75

    school_mixing_type = {'pk': 'age_clustered',
                          'es': 'age_clustered',
                          'ms': 'age_clustered',
                          'hs': 'random',
                          'uv': 'random'}

    T = sc.tic()
    print(f'Making "{popfile}"...')
    popdict = sp.make_population(
                   n = pop_size,
                   rand_seed = seed,
                   generate=True,
                   with_facilities=True,
                   use_two_group_reduction=use_two_group_reduction,
                   average_LTCF_degree=average_LTCF_degree,
                   ltcf_staff_age_min=ltcf_staff_age_min,
                   ltcf_staff_age_max=ltcf_staff_age_max,
                   with_school_types=with_school_types,
                   school_mixing_type=school_mixing_type,
                   average_class_size=average_class_size,
                   inter_grade_mixing=inter_grade_mixing,
                   average_student_teacher_ratio=average_student_teacher_ratio,
                   average_teacher_teacher_degree=average_teacher_teacher_degree,
                   teacher_age_min=teacher_age_min,
                   teacher_age_max=teacher_age_max,
                   with_non_teaching_staff=with_non_teaching_staff,
                   average_student_all_staff_ratio=average_student_all_staff_ratio,
                   average_additional_staff_degree=average_additional_staff_degree,
                   staff_age_min=staff_age_min,
                   staff_age_max=staff_age_max,
                   )

    if do_save:
        sc.saveobj(popfile, popdict)

    sc.toc(T)
    print(f'Done, saved to {popfile}')
    return popdict


if __name__ == '__main__':

    seed = 3
    pop_size = 20e3
    fn = 'testpop4.pop'
    popdict = cache_populations(seed, pop_size, fn)
