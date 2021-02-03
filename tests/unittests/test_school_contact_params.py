"""
Tests to cover the below school related parameters
average_student_teacher_ratio,
average_teacher_teacher_degree,
average_student_all_staff_ratio,
average_additional_staff_degree,
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import synthpops as sp
pars = dict(
    n=2e4,
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

do_show = False
do_save = True
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
    contacts = get_contact_counts(pop.popdict,
                                  "average_student_all_staff_ratio",
                                  average_student_all_staff_ratio,
                                  do_show, do_save)
    print(f"expected:{average_student_all_staff_ratio}")
    average_student_all_staff_count = contacts['sc_staff']['sc_student'] + contacts['sc_teacher']['sc_student']
    actual_mean = np.average(average_student_all_staff_count)
    actual_std = np.std(contacts['sc_staff']['sc_student'])
    print(f"actual student_all_staff_ratio:{round(actual_mean,2)}")
    assert average_student_all_staff_ratio - actual_std <= actual_mean <= average_student_all_staff_ratio + actual_std, \
        f"value is way off from:{np.round(average_student_all_staff_ratio - actual_std, 2)} " \
        f"to {np.round(average_student_all_staff_ratio + actual_std, 2)}"


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
                                  do_show, do_save)
    print(f"expected:{average_additional_staff_degree}")
    actual_mean = np.average(contacts['sc_staff']['all'])
    actual_std = np.std(contacts['sc_staff']['all'])
    print(f"actual staff_degree:{round(actual_mean,2)}")
    assert average_additional_staff_degree - actual_std <= actual_mean <= average_additional_staff_degree + actual_std, \
        f"value is way off from:{np.round(average_additional_staff_degree - actual_std, 2)} " \
        f"to {np.round(average_additional_staff_degree + actual_std, 2)}"


@pytest.mark.parametrize("average_student_teacher_ratio", [20, 30, 40])
def test_average_student_teacher_ratio(average_student_teacher_ratio):
    """
    Test case for average_student_teacher_ratio by taking average of student contacts per teacher
    Args:
        average_student_teacher_ratio: The average number of students per teacher

    Returns:

    """
    pars["average_student_teacher_ratio"] = average_student_teacher_ratio
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_student_teacher_ratio",
                                  average_student_teacher_ratio,
                                  do_show, do_save)
    print(f"expected:{average_student_teacher_ratio}")
    actual_mean = np.average(contacts['sc_teacher']['sc_student'])
    actual_std = np.std(contacts['sc_teacher']['sc_student'])
    print(f"actual student_teacher_ratio:{round(actual_mean,2)}")
    assert average_student_teacher_ratio-actual_std <= actual_mean <= average_student_teacher_ratio+actual_std, \
        f"value is way off from:{np.round(average_student_teacher_ratio-actual_std,2)} " \
        f"to {np.round(average_student_teacher_ratio+actual_std,2)}"


@pytest.mark.parametrize("average_teacher_teacher_degree", [1, 5, 8])
def test_average_teacher_teacher_degree(average_teacher_teacher_degree):
    """
    Test case for average_teacher_teacher_degree by taking average of teachers' contacts per teacher
    Args:
        average_teacher_teacher_degree: The average number of contacts per teacher with other teachers

    Returns:

    """
    pars["average_teacher_teacher_degree"] = average_teacher_teacher_degree
    pop = sp.Pop(**pars)
    contacts = get_contact_counts(pop.popdict,
                                  "average_teacher_teacher_degree",
                                  average_teacher_teacher_degree,
                                  do_show, do_save)
    print(f"expected:{average_teacher_teacher_degree}")
    actual_mean = np.average(contacts['sc_teacher']['sc_teacher'])
    actual_std = np.std(contacts['sc_teacher']['sc_teacher'])
    print(f"actual teacher_teacher_degree:{round(actual_mean,2)}")
    assert average_teacher_teacher_degree-actual_std <= actual_mean <= average_teacher_teacher_degree+actual_std, \
        f"value is way off from:{np.round(average_teacher_teacher_degree-actual_std,2)} " \
        f"to {np.round(average_teacher_teacher_degree+actual_std,2)}"


def get_contact_counts(popdict, varname, varvalue, do_show=do_show, do_save=do_save):
    """
    Helper method to get contact counts for teachers, students and staffs in the popdict
    Args:
        popdict: popdict of a Pop object
        varname: variable name used for plotting to identify the test cases
        varvalue: variable value used for plotting to identify the test cases
        do_show: whether to plot the count distribution or not

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
            # counter_switcher is a case-switch selector for contact counts by type
            count_switcher = {
                'sc_teacher': len([c for c in person["contacts"]["S"] if popdict[c]['scid'] == person['scid'] and popdict[c]['sc_teacher']]),
                'sc_student': len([c for c in person["contacts"]["S"] if popdict[c]['scid'] == person['scid'] and popdict[c]['sc_student']]),
                'sc_staff': len([c for c in person["contacts"]["S"] if popdict[c]['scid'] == person['scid'] and popdict[c]['sc_staff']]),
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
    fig.suptitle(varname + "=" + str(varvalue), fontsize=14)
    for ax, counter in zip(axes.flatten(), list(itertools.product(keys, categories))):
        ax.hist(contact_counter[counter[0]][counter[1]])
        ax.set(title=f'{counter[0]} to {counter[1]}')
        # ax.set_xlabel("num_contacts")
    if do_show:
        plt.show()
    if do_save:
        testfolder = os.path.splitext(os.path.basename(__file__))[0]
        os.makedirs(testfolder, exist_ok=True)
        plt.savefig(f"{testfolder}/{varname}_{str(varvalue)}.png")
    return contact_counter

if __name__ == "__main__":
   pytest.main(['-v', __file__])