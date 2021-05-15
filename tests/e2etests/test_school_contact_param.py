"""
Tests to cover the below school related parameters
average_student_teacher_ratio,
average_teacher_teacher_degree,
average_student_all_staff_ratio,
average_additional_staff_degree,
average_class_size
"""
from scipy import stats
import numpy as np
import collections
import itertools
import matplotlib.pyplot as plt
import os
import pytest
import pathlib
import synthpops as sp
import sciris as sc
from setup_e2e import get_fig_dir

pars = dict(
    n                       = 15e3,
    rand_seed               = 123,
    with_non_teaching_staff = 1
)


@pytest.mark.parametrize("school_mixing_type", ['random', 'age_clustered', 'age_and_class_clustered'])
@pytest.mark.parametrize("average_class_size", [10, 50])
def test_average_class_size(average_class_size, school_mixing_type, do_show, do_save, get_fig_dir, quantiles=None):
    """
    Test case to check average_class_size by taking average of student-student contacts

    Args:
        average_class_size: The average classroom size.

    Returns:
        None
    """
    sp.logger.info(f"Test average_class_size: {average_class_size}.")
    testpars = dict(
        average_class_size            = average_class_size,
        average_student_teacher_ratio = 20,  # DM: note that this parameter will overide the average class size parameter when school mixing types are something other than random or undefined (which defaults to random) --- method refactor work for schools will clarify these relationships
        with_school_types             = 1,
        school_mixing_type            = school_mixing_type,
    )
    pop = sp.Pop(**pars, **testpars)
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir)

    contacts = sp.get_contact_counts_by_layer(pop.popdict, with_layer_ids=1)[0]
    title = f"Average Class Size = {average_class_size} {pop.school_pars.school_mixing_type}"
    plotting_kwargs.append("title_prefix", title)
    plotting_kwargs.append("figname", f"contact_average_class_size_{average_class_size}_{pop.school_pars.school_mixing_type}")
    sp.plot_contact_counts(contacts, **plotting_kwargs)

    counts = []
    if not pop.school_pars.with_school_types:
        counts.extend(contacts['sc_student']['all'])
        counts.extend(contacts['sc_teacher']['all'])
        counts.extend(contacts['sc_staff']['all'])

    elif (pop.school_pars.with_school_types) & (pop.school_pars.school_mixing_type == 'random'):
        counts.extend(contacts['sc_student']['all'])
        counts.extend(contacts['sc_teacher']['all'])

    elif (pop.school_pars.with_school_types) & (pop.school_pars.school_mixing_type == 'age_clustered'):
        counts.extend(contacts['sc_student']['sc_student'])

    elif (pop.school_pars.with_school_types) & (pop.school_pars.school_mixing_type == 'age_and_class_clustered'):
        counts.extend(contacts['sc_student']['sc_student'])
        if pop.school_pars.average_class_size < pop.school_pars.average_student_teacher_ratio:
            average_class_size = pop.school_pars.average_student_teacher_ratio

    sp.check_poisson(actual=counts, expected=average_class_size, label='average_class_size', check='dist')
    # visual check with scipy.stats.probplot -- temporary, just to show that the null hypothesis should pass here for the distribution
    fig, ax = plt.subplots(1, 1)
    res = stats.probplot(counts, dist=stats.poisson, sparams=(average_class_size, ), plot=ax)
    ax.set_title(f'Probability Plot for {title}')
    if do_show:
        plt.show()
    plt.close()
    return


@pytest.mark.parametrize("average_additional_staff_degree", [20, 40])
def test_average_additional_staff_degree(average_additional_staff_degree, do_show, do_save, get_fig_dir, threshold=2):
    """
    Test case to check average_additional_staff_degree by taking average of all contacts per staff

    Args:
        average_additional_staff_degree: The average number of contacts per additional non teaching staff in schools

    Returns:
        None
    """
    # note this must be greater than default average_student_all_staff_ratio (20)
    testpars = dict(
        average_additional_staff_degree = average_additional_staff_degree,
        with_school_types = 1,
    )
    pop = sp.Pop(**pars, **testpars)
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir)
    contacts = sp.get_contact_counts_by_layer(pop.popdict, with_layer_ids=1)[0]
    plotting_kwargs.append("title_prefix", f"Average Additional Staff Degree = {average_additional_staff_degree}")
    plotting_kwargs.append("figname", f"contact_average_additional_staff_degree_{average_additional_staff_degree}")
    sp.plot_contact_counts(contacts, **plotting_kwargs)
    counts = contacts['sc_staff']['all']
    sp.check_normal(actual=counts, expected=average_additional_staff_degree, label='staff degree', check='mean')
    return


@pytest.mark.parametrize("average_student_teacher_ratio", [20, 40])
def test_average_student_teacher_ratio(average_student_teacher_ratio, do_show, do_save, get_fig_dir, threshold=2):
    """
    Test case for average_student_teacher_ratio by taking average of student contacts per teacher

    Args:
        average_student_teacher_ratio: The average number of students per teacher

    Returns:
        None
    """
    testpars = dict(
        average_student_teacher_ratio = average_student_teacher_ratio,
    )
    pop = sp.Pop(**pars, **testpars)
    ratios = get_teacher_staff_ratio(pop.popdict,
                                     "average_student_teacher_ratio",
                                     average_student_teacher_ratio,
                                     do_show, do_save, get_fig_dir)
    sp.check_normal(actual=ratios, expected=average_student_teacher_ratio, label='average_student_teacher_ratio', check='mean')
    return


@pytest.mark.parametrize("average_student_all_staff_ratio", [10, 20])
def test_student_all_staff_ratio(average_student_all_staff_ratio, do_show, do_save, get_fig_dir, threshold=2):
    """
    Test case to check average_student_all_staff_ratio by taking average of students contacts from teachers and staff

    Args:
        average_student_all_staff_ratio: The average number of students per staff members at school
        (including both teachers and non teachers)

    Returns:
        None
    """
    testpars = dict(
        average_student_all_staff_ratio = average_student_all_staff_ratio,
    )
    pop = sp.Pop(**pars, **testpars)
    ratios = get_teacher_staff_ratio(pop.popdict,
                                     "average_student_all_staff_ratio",
                                     average_student_all_staff_ratio,
                                     do_show, do_save, get_fig_dir)
    sp.check_normal(actual=ratios, expected=average_student_all_staff_ratio, label='average_student_all_staff_ratio', check='mean')
    return


@pytest.mark.parametrize("average_teacher_teacher_degree", [1, 8])
def test_average_teacher_teacher_degree(average_teacher_teacher_degree, do_show, do_save, get_fig_dir, threshold=2):
    """
    Test case for average_teacher_teacher_degree by taking average of teachers' contacts per teacher

    Args:
        average_teacher_teacher_degree: The average number of contacts per teacher with other teachers

    Returns:
        None
    """
    testpars = dict(
        average_teacher_teacher_degree = average_teacher_teacher_degree,
        with_school_types = 1,
        school_mixing_type = {'pk': 'age_and_class_clustered', # average_teacher_teacher_degree will not be used in school_mixing_type == 'random' scenario
                              'es': 'age_and_class_clustered',
                              'ms': 'age_and_class_clustered',
                              'hs': 'age_clustered', 'uv': 'age_clustered'}
    )
    pop = sp.Pop(**pars, **testpars)
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir)
    contacts = sp.get_contact_counts_by_layer(pop.popdict, with_layer_ids=1)[0]
    plotting_kwargs.append("title_prefix", f"Average Teacher-Teacher Degree = {average_teacher_teacher_degree}")
    plotting_kwargs.append("figname", f"contact_average_teacher_teacher_degree_{average_teacher_teacher_degree}")
    sp.plot_contact_counts(contacts, **plotting_kwargs)
    counts = contacts['sc_teacher']['sc_teacher']
    sp.check_normal(actual=counts, expected=average_teacher_teacher_degree, label='teacher degree', check='mean')
    return


def get_teacher_staff_ratio(popdict, varname, varvalue, do_show, do_save, fig_dir):
    """
    Method to generate the student_teacher and student_all_staff ratio from popdict

    Args:
        popdict  : popdict of a Pop object
        varname  : variable name used for identifying the test cases, must be
                 average_student_teacher_ratio or average_student_all_staff_ratio
        varvalue : variable value used for plotting to identify the test cases
        do_show  : whether to plot the count distribution or not
        do_save  : whether to save the plot or not
        fig_dir  : subfolder name (under current run directory) for saving the plots

    Returns:
        average and std value of the varname arg
    """
    school_students = collections.defaultdict(list)
    school_teachers = collections.defaultdict(list)
    school_staffs = collections.defaultdict(list)
    # Count the students, teachers, staff group by scid
    for uid, person in popdict.items():
        if person['sc_student']:
            school_students[person['scid']].append(uid)
        elif person['sc_teacher']:
            school_teachers[person['scid']].append(uid)
        elif person['sc_staff']:
            school_staffs[person['scid']].append(uid)

    school_view = {}
    # generate student_teacher_ratio and student_all_staff_ratio group by scid
    for i in school_students.keys():
        school_view[i] = {'student_teacher_ratio': len(school_students[i]) / len(school_teachers[i]),
                          'student_all_staff_ratio': len(school_students[i]) / (
                                      len(school_staffs[i]) + len(school_teachers[i]))}

    # based on the varname, plot and return and mean/std
    if varname == "average_student_teacher_ratio":
        ratio = [v['student_teacher_ratio'] for v in school_view.values()]
    elif varname == "average_student_all_staff_ratio":
        ratio = [v['student_all_staff_ratio'] for v in school_view.values()]

    # draw a simple histogram for distribution of ratio
    if do_show or do_save:
        plt.title(f"{varname} = {str(varvalue)}", fontsize=14)
        plt.hist([round(i) for i in ratio])
        if do_show:
            plt.show()
        if do_save:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{varname}_{str(varvalue)}.png"))
        plt.close()
    return ratio


if __name__ == "__main__":
    # pytest.main(['-vs', __file__])

    # # run an example in main and plot!
    average_class_size = 30
    school_mixing_type = 'random'
    do_show = 1
    do_save = 0
    test_average_class_size(average_class_size, school_mixing_type, do_show, do_save, get_fig_dir)
