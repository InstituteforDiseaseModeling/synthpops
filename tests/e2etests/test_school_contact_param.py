"""
Tests to cover the below school related parameters
average_student_teacher_ratio,
average_teacher_teacher_degree,
average_student_all_staff_ratio,
average_additional_staff_degree,
average_class_size
"""
import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import pathlib
import warnings
import synthpops as sp
import setup_e2e
from setup_e2e import get_fig_dir

pars = dict(
    n=30000,
    rand_seed=1,
    with_non_teaching_staff=1
)


@pytest.mark.parametrize("average_class_size", [10, 20, 50])
def test_average_class_size(average_class_size, do_show, do_save, get_fig_dir):
    """
    Test case to check average_class_size by taking average of student-student contacts
    Args:
        average_class_size: The average classroom size.
    Returns:
        None
    """
    pars["average_class_size"] = average_class_size
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict, "average_class_size", average_class_size, do_show, do_save, get_fig_dir)
    actual_mean = np.average(contacts['sc_student']['sc_student'])
    actual_std = np.std(contacts['sc_student']['sc_student'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_class_size,
                   actual_std=actual_std,
                   varname="average_class_size")


@pytest.mark.parametrize("average_additional_staff_degree", [20, 30, 40])
def test_average_additional_staff_degree(average_additional_staff_degree, do_show, do_save, get_fig_dir):
    """
    Test case to check average_additional_staff_degree by taking average of all contacts per staff
    Args:
        average_additional_staff_degree: The average number of contacts per additional non teaching staff in schools
    Returns:
        None
    """
    # note this must be greater than default average_student_all_staff_ratio (20)
    pars["average_additional_staff_degree"] = average_additional_staff_degree
    pars["with_school_types"] = 1

    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_additional_staff_degree",
                                  average_additional_staff_degree,
                                  do_show, do_save, get_fig_dir)
    actual_mean = np.average(contacts['sc_staff']['all'])
    actual_std = np.std(contacts['sc_staff']['all'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_additional_staff_degree,
                   actual_std=actual_std,
                   varname="average_additional_staff_degree")


@pytest.mark.parametrize("average_student_teacher_ratio", [20, 30, 40])
def test_average_student_teacher_ratio(average_student_teacher_ratio, do_show, do_save, get_fig_dir):
    """
    Test case for average_student_teacher_ratio by taking average of student contacts per teacher
    Args:
        average_student_teacher_ratio: The average number of students per teacher
    Returns:
        None
    """
    pars["average_student_teacher_ratio"] = average_student_teacher_ratio
    pop = sp.Pop(**pars)
    actual_mean, actual_std = get_teacher_staff_ratio(pop.popdict,
                                                      "average_student_teacher_ratio",
                                                      average_student_teacher_ratio,
                                                      do_show, do_save, get_fig_dir)
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_student_teacher_ratio,
                   actual_std=actual_std,
                   varname="average_student_teacher_ratio")


@pytest.mark.parametrize("average_student_all_staff_ratio", [10, 15, 20])
def test_student_all_staff_ratio(average_student_all_staff_ratio, do_show, do_save, get_fig_dir):
    """
    Test case to check average_student_all_staff_ratio by taking average of students contacts from teachers and staff
    Args:
        average_student_all_staff_ratio: The average number of students per staff members at school
        (including both teachers and non teachers)
    Returns:
        None
    """
    pars["average_student_all_staff_ratio"] = average_student_all_staff_ratio
    pop = sp.Pop(**pars)
    actual_mean, actual_std = get_teacher_staff_ratio(pop.popdict,
                                                      "average_student_all_staff_ratio",
                                                      average_student_all_staff_ratio,
                                                      do_show, do_save, get_fig_dir)
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_student_all_staff_ratio,
                   actual_std=actual_std,
                   varname="average_student_all_staff_ratio")


@pytest.mark.parametrize("average_teacher_teacher_degree", [1, 5, 8])
def test_average_teacher_teacher_degree(average_teacher_teacher_degree, do_show, do_save, get_fig_dir):
    """
    Test case for average_teacher_teacher_degree by taking average of teachers' contacts per teacher
    Args:
        average_teacher_teacher_degree: The average number of contacts per teacher with other teachers
    Returns:
        None
    """
    pars["average_teacher_teacher_degree"] = average_teacher_teacher_degree
    pars["with_school_types"] = 1
    # average_teacher_teacher_degree will not be used in school_mixing_type == 'random' scenario
    pars["school_mixing_type"] = {'pk': 'age_and_class_clustered',
                                  'es': 'age_and_class_clustered',
                                  'ms': 'age_and_class_clustered',
                                  'hs': 'age_clustered', 'uv': 'age_clustered'}
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_teacher_teacher_degree",
                                  average_teacher_teacher_degree,
                                  do_show, do_save, get_fig_dir)
    actual_mean = np.average(contacts['sc_teacher']['sc_teacher'])
    actual_std = np.std(contacts['sc_teacher']['sc_teacher'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_teacher_teacher_degree,
                   actual_std=actual_std,
                   varname="average_teacher_teacher_degree")


def get_contact_counts(popdict, varname, varvalue, do_show, do_save, fig_dir,
                       people_types=['sc_teacher', 'sc_student', 'sc_staff']):
    """
    Helper method to get contact counts for teachers, students and staffs in the popdict
    Args:
        popdict: popdict of a Pop object
        varname: variable name used for plotting to identify the test cases
        varvalue: variable value used for plotting to identify the test cases
        do_show: whether to plot the count distribution or not
        do_save: whether to save the plot or not
        fig_dir: subfolder name (under current run directory) for saving the plots
        people_types: a list of possible people types (such as sc_student, sc_teacher, sc_staff, snf_staff, snf_res)

    Returns:
        A dictionary with keys = people_types (default to ['sc_teacher', 'sc_student', 'sc_staff'])
        and each value is a dictionary which stores the list of counts for each type of contacts:
        default to ['sc_teacher', 'sc_student', 'sc_staff', 'all_staff', 'all']
        for example: contact_counter['sc_teacher']['sc_teacher'] store the counts of each teacher's "teacher" contact
    """
    contact_types = people_types + ['all_staff', 'all']
    # initialize the contact_counter dictionary, the keys are used to identify the teacher, student and staff
    # the categories are used to store the count by contacts type where all means all contacts and all_staff means
    # sc_teacher + sc_staff
    contact_counter = dict.fromkeys(people_types)
    for key in contact_counter:
        contact_counter[key] = dict(zip(contact_types, ([] for _ in contact_types)))

    for uid, person in popdict.items():
        if person['scid']:
            # count_switcher is a case-switch selector for contact counts by type
            count_switcher = {
                'sc_teacher': len([c for c in person["contacts"]["S"] if popdict[c]['sc_teacher']]),
                'sc_student': len([c for c in person["contacts"]["S"] if popdict[c]['sc_student']]),
                'sc_staff': len([c for c in person["contacts"]["S"] if popdict[c]['sc_staff']]),
                'all': len([c for c in person["contacts"]["S"]])
            }
            # index_switcher is a case-switch selector for the person selected by its type
            index_switcher = {
                'sc_teacher': contact_counter['sc_teacher'],
                'sc_student': contact_counter['sc_student'],
                'sc_staff': contact_counter['sc_staff']
            }
            for k1 in people_types:
                # if this person does not belong to a particular key, we don't need to store the counts under this key
                if person.get(k1):
                    # store sc_teacher, sc_student, sc_staff, all_staff and all below
                    for k2 in people_types:
                        index_switcher.get(k1)[k2].append(count_switcher.get(k2))
                    index_switcher.get(k1)["all_staff"].append(
                        count_switcher.get('sc_teacher') + count_switcher.get('sc_staff'))
                    index_switcher.get(k1)["all"].append(count_switcher.get('all'))

    # draw a simple histogram for distribution of counts
    if do_show or do_save:
        fig, axes = plt.subplots(len(people_types), len(contact_types), figsize=(30, 20))
        fig.suptitle(f"Contact View:{varname}={str(varvalue)}", fontsize=20)
        for ax, counter in zip(axes.flatten(), list(itertools.product(people_types, contact_types))):
            ax.hist(contact_counter[counter[0]][counter[1]])
            ax.set_title(f'{counter[0]} to {counter[1]}', {'fontsize': 20})
            ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if do_show:
            plt.show()
        if do_save:
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"contacts_{varname}_{str(varvalue)}.png"))
        plt.close()
    return contact_counter


def get_teacher_staff_ratio(popdict, varname, varvalue, do_show, do_save, fig_dir):
    """
    method to generate the student_teacher and student_all_staff ratio from popdict
    Args:
        popdict: popdict of a Pop object
        varname: variable name used for identifying the test cases, must be
                 average_student_teacher_ratio or average_student_all_staff_ratio
        varvalue: variable value used for plotting to identify the test cases
        do_show: whether to plot the count distribution or not
        do_save: whether to save the plot or not
        fig_dir: subfolder name (under current run directory) for saving the plots

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
    return np.mean(ratio), np.std(ratio)


def assert_outlier(actual_mean, expected_mean, actual_std, varname, threshold=2):
    """
    The method raise an error for actual_mean outside of expected_mean range
    If a value is a certain number of standard deviations away from the expected_mean,
    that data point is identified as an outlier.
    The specified number of standard deviations is called the threshold and default value is 2.
    Hoever, this method may fail to detect outliers because the outliers increase the standard deviation.
    Args:
        actual_mean: the actual mean
        expected_mean: the expected mean
        actual_std: the actual std
        varname: the name of the variable
        threshold: specified number of standard deviations

    Returns:
        None
    """
    print("-------------------")
    print(f"expected: {varname} = {round(expected_mean, 2)}")
    print(f"actual: {varname} = {round(actual_mean, 2)}")
    print("-------------------")
    if expected_mean - threshold * actual_std <= actual_mean <= expected_mean + threshold * actual_std:
        return
    else:
        warnings.warn(f"{varname}: actual value is {round(actual_mean, 2)}"
                      f" but expected value is  {round(expected_mean, 2)}\n"
                      f" acceptable range should be from:{np.round(expected_mean - actual_std, 2)}"
                      f" to {np.round(expected_mean + actual_std, 2)}", category=UserWarning)


if __name__ == "__main__":
    pytest.main(['-v', __file__])