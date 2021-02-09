"""
Tests to cover the below school related parameters
average_student_teacher_ratio,
average_teacher_teacher_degree,
average_student_all_staff_ratio,
average_additional_staff_degree,
"""
import collections
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import synthpops as sp
pars = dict(
    n=7000,
    rand_seed=1,
    max_contacts=None,
    country_location='usa',
    state_location='Washington',
    location='seattle_metro',
    use_default=True,
    with_non_teaching_staff=1,
    with_school_types=1,
    school_mixing_type={'pk': 'age_and_class_clustered', 'es': 'age_clustered', 'ms': 'age_clustered', 'hs': 'random', 'uv': 'random'},
    average_class_size=20,
    inter_grade_mixing=0.1
)

fig_dir = os.path.join(os.getcwd(), 'test_school_contact_params')
do_show = False
do_save = False


@pytest.mark.parametrize("average_class_size", [50, 30, 20])
def test_average_class_size(average_class_size):
    """
    Test case to check average_class_size by taking average of student-student contacts
    Args:
        average_class_size: The average classroom size.
    Returns:
        None
    """
    pars["average_class_size"] = average_class_size
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict, "average_class_size", average_class_size, do_show, do_save, fig_dir)
    actual_mean = np.average(contacts['sc_student']['sc_student'])
    actual_std = np.std(contacts['sc_student']['sc_student'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_class_size,
                   actual_std=actual_std,
                   varname="average_class_size")


@pytest.mark.parametrize("average_additional_staff_degree", [1, 5, 10])
def test_average_additional_staff_degree(average_additional_staff_degree):
    """
    Test case to check average_additional_staff_degree by taking average of all contacts per staff
    Args:
        average_additional_staff_degree: The average number of contacts per additional non teaching staff in schools
    Returns:
        None
    """
    pars["average_additional_staff_degree"] = average_additional_staff_degree
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_additional_staff_degree",
                                  average_additional_staff_degree,
                                  do_show, do_save, fig_dir)
    actual_mean = np.average(contacts['sc_staff']['all'])
    actual_std = np.std(contacts['sc_staff']['all'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_additional_staff_degree,
                   actual_std=actual_std,
                   varname="average_additional_staff_degree")


@pytest.mark.parametrize("average_student_teacher_ratio", [20, 30, 40])
def test_average_student_teacher_ratio(average_student_teacher_ratio):
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
                                                      do_show, do_save, fig_dir)
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_student_teacher_ratio,
                   actual_std=actual_std,
                   varname="average_student_teacher_ratio")


@pytest.mark.parametrize("average_student_all_staff_ratio", [10, 15, 20])
def test_student_all_staff_ratio(average_student_all_staff_ratio):
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
                                                      do_show, do_save, fig_dir)
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_student_all_staff_ratio,
                   actual_std=actual_std,
                   varname="average_student_all_staff_ratio")


@pytest.mark.parametrize("average_teacher_teacher_degree", [1, 5, 8])
def test_average_teacher_teacher_degree(average_teacher_teacher_degree):
    """
    Test case for average_teacher_teacher_degree by taking average of teachers' contacts per teacher
    Args:
        average_teacher_teacher_degree: The average number of contacts per teacher with other teachers
    Returns:
        None
    """
    pars["average_teacher_teacher_degree"] = average_teacher_teacher_degree
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_teacher_teacher_degree",
                                  average_teacher_teacher_degree,
                                  do_show, do_save, fig_dir)
    actual_mean = np.average(contacts['sc_teacher']['sc_teacher'])
    actual_std = np.std(contacts['sc_teacher']['sc_teacher'])
    assert_outlier(actual_mean=actual_mean,
                   expected_mean=average_teacher_teacher_degree,
                   actual_std=actual_std,
                   varname="average_teacher_teacher_degree")


def get_contact_counts(popdict, varname, varvalue, do_show, do_save, fig_dir):
    """
    Helper method to get contact counts for teachers, students and staffs in the popdict
    Args:
        popdict: popdict of a Pop object
        varname: variable name used for plotting to identify the test cases
        varvalue: variable value used for plotting to identify the test cases
        do_show: whether to plot the count distribution or not
        do_save: whether to save the plot or not
        fig_dir: subfolder name (under current run directory) for saving the plots

    Returns:
        A dictionary with keys = ['sc_teacher', 'sc_student', 'sc_staff']
        and each value is a dictionary of categories which stores the count representing these 5 categories of contacts:
        ['sc_teacher', 'sc_student', 'sc_staff', 'all_staff', 'all']
    """
    keys = ['sc_teacher', 'sc_student', 'sc_staff']
    categories = ['sc_teacher', 'sc_student', 'sc_staff', 'all_staff', 'all']
    # initialize the contact_counter dictionary, the keys are used to identify the teacher, student and staff
    # the categories are used to store the count by contacts type where all means all contacts and all_staff means
    # sc_teacher + sc_staff
    contact_counter = dict.fromkeys(keys)
    for key in contact_counter:
        contact_counter[key] = dict(zip(categories, ([] for _ in categories)))

    for uid, person in popdict.items():
        if person['scid']:
            # count_switcher is a case-switch selector for contact counts by type
            count_switcher = {
                'sc_teacher': len([c for c in person["contacts"]["S"] if popdict[c]['sc_teacher']]),
                'sc_student': len([c for c in person["contacts"]["S"] if popdict[c]['sc_student']]),
                'sc_staff': len([c for c in person["contacts"]["S"] if popdict[c]['sc_staff']]),
                'all': len([c for c in person["contacts"]["S"] if popdict[c]['scid'] == person['scid']])
            }
            # index_switcher is a case-switch selector for the person selected by its type
            index_switcher = {
                'sc_teacher': contact_counter['sc_teacher'],
                'sc_student': contact_counter['sc_student'],
                'sc_staff': contact_counter['sc_staff']
            }
            for k1 in keys:
                # if this person does not belong to a particular key, we don't need to store the counts under this key
                if not person[k1]:
                    continue
                # store sc_teacher, sc_student, sc_staff, all_staff and all below
                for k2 in keys:
                    index_switcher.get(k1)[k2].append(count_switcher.get(k2))
                index_switcher.get(k1)["all_staff"].append(count_switcher.get('sc_teacher') + count_switcher.get('sc_staff'))
                index_switcher.get(k1)["all"].append(count_switcher.get('all'))

    # draw a simple histogram for distribution of counts
    fig, axes = plt.subplots(len(keys), len(categories), figsize=(30, 20))
    fig.suptitle(f"Contact View:{varname}={str(varvalue)}", fontsize=20)
    for ax, counter in zip(axes.flatten(), list(itertools.product(keys, categories))):
        ax.hist(contact_counter[counter[0]][counter[1]])
        ax.set(title=f'{counter[0]} to {counter[1]}')
        # ax.set_xlabel("num_contacts")
    if do_show:
        plt.show()
    if do_save:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/contacts_{varname}_{str(varvalue)}.png")
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
    students = collections.defaultdict(list)
    teachers = collections.defaultdict(list)
    staffs = collections.defaultdict(list)
    # Count the students, teachers, staff group by scid
    for uid, person in popdict.items():
        if person['sc_student']:
            students[person['scid']].append(uid)
        elif person['sc_teacher']:
            teachers[person['scid']].append(uid)
        elif person['sc_staff']:
            staffs[person['scid']].append(uid)

    school_view = {}
    # generate student_teacher_ratio and student_all_staff_ratio group by scid
    for i in students.keys():
        school_view[i] = {'student_teacher_ratio': len(students[i])/len(teachers[i]),
                          'student_all_staff_ratio': len(students[i])/(len(staffs[i]) + len(teachers[i]))}

    # based on the varname, plot and return and mean/std
    if varname == "average_student_teacher_ratio":
        ratio = [v['student_teacher_ratio'] for v in school_view.values()]
    elif varname == "average_student_all_staff_ratio":
        ratio = [v['student_all_staff_ratio'] for v in school_view.values()]

    # draw a simple histogram for distribution of ratio
    plt.title(varname + "=" + str(varvalue), fontsize=14)
    plt.hist([round(i) for i in ratio])
    if do_show:
        plt.show()
    if do_save:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{varname}_{str(varvalue)}.png")
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
    print(f"expected: {varname} = {round(expected_mean, 2)}")
    print(f"actual: {varname} = {round(actual_mean, 2)}")
    assert expected_mean - threshold * actual_std <= actual_mean <= expected_mean + threshold * actual_std, \
        f"value is way off from:{np.round(expected_mean - actual_std, 2)} " \
        f"to {np.round(expected_mean + actual_std, 2)}"

if __name__ == "__main__":
   pytest.main(['-v', __file__])