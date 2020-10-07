'''
Uitilities for test_school_staff (and possibly other functions)
'''

import synthpops as sp
import sciris as sc
import inspect
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import stats


def runpop(resultdir, actual_vals, testprefix, method):
    """
    Run any method which creates a population and write args and population to
    file "{resultdir}/{testprefix}.json". The method must be a method which
    returns a population. Write the population file to
    "{resultdir}/{testprefix}.config.json"

    Args:
      resultdir (str)    : result directory
      actual_vals (dict) : a dictionary with param name and param value
      testprefix (str)   : test prefix to generate file name

    Returns:
        Population dictionary
    """
    os.makedirs(resultdir, exist_ok=True)
    params = {}
    args = inspect.getfullargspec(method).args
    for arg in args:
        params[arg] = inspect.signature(method).parameters[arg].default
    for name in actual_vals:
        if name in params.keys():
            params[name] = actual_vals[name]
    sc.savejson(os.path.join(resultdir, f"{testprefix}.config.json"), params, indent=2)
    pop = method(**params)
    sc.savejson(os.path.join(resultdir, f"{testprefix}_pop.json"), pop, indent=2)
    return pop


def copy_input(sourcedir, resultdir, subdir_level):
    """
    Copy files to the target datadir up to the subdir level.

    Args:
        sourcedir (str): source directory
        resultdir (str): result directory
        subdir_level (str): sub-directory

    Returns:
        None
    """

    # copy all files to datadir except the ignored files
    ignorepatterns = shutil.ignore_patterns("*contact_networks*",
                                            "*contact_networks_facilities*",
                                            "*New_York*",
                                            "*Oregon*")
    shutil.copytree(sourcedir, os.path.join(resultdir, subdir_level), ignore=ignorepatterns)


def check_teacher_staff_ratio(pop, datadir, test_prefix, average_student_teacher_ratio, average_student_all_staff_ratio,
                              err_margin=0):
    """
    Check if generated population matches average_student_teacher_ratio and
    average_student_all_staff_ratio.

    Args:
        pop (dict): population dictionary
        datadir (str): data directory
        test_prefix (str): prefix str for test
        average_student_teacher_ratio (float)   : The average number of students per teacher.
        average_student_all_staff_ratio (float) : The average number of students per staff members at school (including both teachers and non teachers).
        err_margin (float): error margin

    Returns:
        Average number of students per teacher and average number of students per all staff in the input population.
    """
    i = 0
    school = {}
    for p in pop.values():
        if p["scid"] is not None:
            row = {"scid": p["scid"],
                   "student": 0 if p["sc_student"] is None else p["sc_student"],
                   "teacher": 0 if p["sc_teacher"] is None else p["sc_teacher"],
                   "staff": 0 if p["sc_staff"] is None else p["sc_staff"]}
            school[i] = row
            i += 1
    df_school = pd.DataFrame.from_dict(school).transpose()
    result = df_school.groupby('scid', as_index=False)[['student', 'teacher', 'staff']].agg(lambda x: sum(x))
    result["teacher_ratio"] = result["student"] / (result["teacher"])
    result["allstaff_ratio"] = result["student"] / (result["teacher"] + result["staff"])
    print(result.head(20))
    result.to_csv(os.path.join(os.path.dirname(datadir), f"{test_prefix}_{len(pop)}.csv"), index=False)

    # check for 0 staff/teacher case to see if it is dues to school size being too small
    zero_teacher_case = result.query('teacher == 0 & student > @average_student_teacher_ratio')
    assert (len(zero_teacher_case) == 0), \
        f"All schools with more students than the student teacher ratio should have at least one teacher. {len(zero_teacher_case)} did not."
    zero_staff_case = result.query('staff == 0 & student > @average_student_all_staff_ratio')
    assert (len(zero_staff_case) == 0), \
        f"All schools with more students than the student staff ratio: {average_student_all_staff_ratio} should have at least 1 staff. {len(zero_staff_case)} did not."

    # exclude 0 teacher if size is too small
    result = result[result.teacher > 0][result.staff > 0]

    # exclude student size less than 3*average_student_all_staff_ratio
    # 3 is an experimental number and a more relaxed margin to prevent small schools to impact
    # the average_student_all_staff_ratio as they tend to be larger than average
    # in general, average across schools must match input
    actual_teacher_ratio = result[result["student"] > 3 * average_student_teacher_ratio]["teacher_ratio"].mean()
    print(f"actual average student teacher ratio (ignore small size schools):{actual_teacher_ratio}")
    assert (int(average_student_teacher_ratio + err_margin) >= actual_teacher_ratio >= int(
        average_student_teacher_ratio - err_margin)), \
        f"teacher ratio: expected: {average_student_teacher_ratio} actual: {actual_teacher_ratio}"
    actual_staff_ratio = result[result["student"] > 3 * average_student_teacher_ratio]["allstaff_ratio"].mean()
    print(f"actual average student all staff ratio (ignore small size schools):{actual_staff_ratio}")
    assert (int(average_student_all_staff_ratio + err_margin) >= actual_staff_ratio >= int(
        average_student_all_staff_ratio - err_margin)), \
        f"all staff ratio expected: {average_student_all_staff_ratio} actual: {actual_staff_ratio}"
    return result


def plot_array(expected, actual=None, names=None, datadir=None, testprefix="test", do_close=True, expect_label='expected', value_text=False):
    """
    Plot histogram on a sorted array based by names. If names not provided the
    order will be used. If actual data is not provided, plot only the expected values.

    Args:
        expected (np.ndarray): Array of expected values.
        actual (np.ndarray): Array of actual values.
        names ()
        datadir (str)

    """
    fig, ax = plt.subplots(1, 1)
    font = {
            'weight': 'bold',
            'size': 10
            }
    # plt.rc('font', **font)
    # try:
    mplt.rcParams['font.family'] = 'Roboto Condensed'
    # except:
        # pass
    title = testprefix if actual is None else f"Comparison for {testprefix}"
    # plt.title(title)
    ax.set_title(title)
    names = np.arange(len(expected)) if names is None else names
    print('name', names, len(names))
    print('expected', expected, len(expected), type(expected))
    print('actual', actual)
    ax.hist(x=names, histtype='bar', weights=expected, label=expect_label, bins=len(expected), color='steelblue')
    if actual is not None:
        arr = ax.hist(x=names, histtype='step', lw=3, weights=actual, label='actual', bins=len(actual), color='salmon')
        if value_text:
            #display values
            for i in range(len(actual)):
                plt.text(arr[1][i], arr[0][i], str(round(arr[0][i], 3)))
    ax.legend(loc='upper right')

    if datadir:
        plt.savefig(os.path.join(datadir, f"{testprefix}_graph.png"), format="png")
    if do_close:
        plt.close()
    else:
        plt.show()
    return


def check_age_distribution(pop,
                           n,
                           datadir,
                           figdir,
                           location=None,
                           state_location=None,
                           country_location=None,
                           file_path=None,
                           use_default=False,
                           test_prefix="test",
                           skip_stat_check=False,
                           do_close=True):
    """
    construct histogram from expected age distribution
    compare with the actual generated data
    """
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    actual_age_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    for p in pop.values():
       actual_age_dist[ageindex[p["age"]]] += 1

    # un-normalized data
    # expected_values = np.array(list(age_dist.values())) * n
    # actual_values = np.array(list(actual_age_dist.values()))
    # normalized
    expected_values = np.array(list(age_dist.values()))
    actual_values = np.array(list(sp.norm_dic(actual_age_dist).values()))
    names = np.array([i[0] for i in brackets.values()])
    plot_array(expected_values, actual_values, names, figdir,  "age_distribution_" + test_prefix, do_close=do_close)
    if not skip_stat_check:
        statistic_test(expected_values, actual_values, test="x", comments="age distribution check")


def check_enrollment_distribution(pop,
                                  n,
                                  datadir,
                                  figdir,
                                  location=None,
                                  state_location=None,
                                  country_location=None,
                                  file_path=None,
                                  use_default=False,
                                  test_prefix="test",
                                  skip_stat_check=False,
                                  do_close=True,
                                  plot_only=False):
    """
    Compute the statistic on expected enrollment-age distribution and compare with actual distribution
    check zero enrollment bins to make sure there is nothing generated
    """
    expected_dist = sp.get_school_enrollment_rates(datadir, location, state_location, country_location, file_path,
                                                   use_default)
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)

    # get actual school enrollment by age
    actual_pool = []
    actual_dist = dict.fromkeys(list(range(0, 101)), 0)
    for p in pop.values():
        if p["scid"] is not None and p["sc_student"] is not None:
            actual_dist[p["age"]] += 1
            actual_pool.append(p["age"])

    # adjust expected enrollment percentage by age brackets
    expected_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    actual_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    scaled_dist = dict.fromkeys(list(range(0, 101)), 0)
    for i in age_dist:
        for j in brackets[i]:
            scaled_dist[j] = (expected_dist[j] * n * age_dist[i]) / len(brackets[i])
            expected_combined_dist[i] += scaled_dist[j]
            actual_combined_dist[i] += actual_dist[j]

    # construct expected pool based on adjusted distribution
    expected_pool = []
    for key in scaled_dist:
        for i in range(0, int(scaled_dist[key])):
            expected_pool.append(key)

    print(f"total enroll expected :{int(sum(scaled_dist.values()))}")
    print(f"total enroll actual :{sum(actual_dist.values())}")

    # make sure results are sorted by key
    # scaled_dist_dist = dict(sorted(scaled_dist.items()))
    actual_dist = dict(sorted(actual_dist.items()))

    expected_values = np.array(list(scaled_dist.values()))
    actual_values = np.array(list(actual_dist.values()))
    expected_combined_values = np.array(list(expected_combined_dist.values()))
    actual_combined_values = np.array(list(actual_combined_dist.values()))

    # uncomment below if you need to plot and check data
    plot_array(expected_values, actual_values, None, figdir, "enrollment_" + test_prefix, do_close=do_close)
    plot_array(expected_combined_values, actual_combined_values, np.array([i[0] for i in brackets.values()]),
               figdir, "enrollment_combined_" + test_prefix, do_close=do_close)
    if plot_only:
        return
    np.savetxt(os.path.join(os.path.dirname(datadir), f"{test_prefix}_expected.csv"), expected_values, delimiter=",")
    np.savetxt(os.path.join(os.path.dirname(datadir), f"{test_prefix}_actual.csv"), actual_values, delimiter=",")

    # check for expected 0 count bins
    # if expected enrollment is 0, actual enrollment must be 0
    # if the expected enrollment is greater than threshold, actual enrollment should not be zero
    # here we use tentative threshold 9 meaning if we expected 10+ enrollment and actually
    # generate 0, we should investigate why
    threshold = 9
    assert np.sum(actual_values[expected_values == 0]) == 0, \
        f"expected enrollment should be 0 for these age bins: " \
        f"{str(np.where((expected_values == 0) & (actual_values != 0)))}"

    assert len(actual_values[np.where((expected_values > threshold) & (actual_values == 0))]) == 0, \
        f"actual enrollment should not be 0 for these age bins: " \
        f"{str(np.where((expected_values > threshold) & (actual_values == 0)))}"

    # if bin count less than threshold use range check to allow up
    # not exceeding up to 2*threshold
    i = np.where((expected_values <= threshold) & (expected_values > 0))
    u = expected_values[i] + threshold  # upper bound
    l = np.zeros(len(expected_values[i]))  # lower bound can be 0
    assert (sum(l <= actual_values[i]) == len(actual_values[i]) and sum(actual_values[i] <= u) == len(
        actual_values[i])), \
        f"results show too much difference:\n" \
        f"expected:{expected_values[i]} \n actual:{actual_values[i]} \n" \
        f"please check these age bins: {i}"

    # check if pool looks right
    # h, bins = np.histogram(np.array(expected_pool), bins=100)
    # h, bins = np.histogram(np.array(actual_pool), bins=100)
    # plt.bar(bins[:-1],h,width=1)
    # plt.show()

    if not skip_stat_check:
        statistic_test(expected_pool, actual_pool, test="ks", comments="enrollment distribution check")
    # todo: theoretically this should work, however does not pass in our example
    # statistic_test(actual_combined_values[expected_combined_values > 0],
    # expected_combined_values[expected_combined_values > 0], test="x")


def statistic_test(expected, actual, test="ks", comments=""):
    print(comments)
    if test == "ks":
        print("use Kolmogorov-Smirnov statistic to check actual distribution")
        s, p = stats.ks_2samp(expected, actual)
        print(f"KS statistics: {s} pvalue:{p}")
    elif test == "x":
        print("use Chi-square statistic")
        s, p = stats.chisquare(actual, f_exp=expected, ddof=0, axis=0)
        print(f"chi square statistics: {s} pvalue:{p}")

    assert p > 0.05, f"Under the null hypothesis the expected/actual distributions are identical." \
                     f" If statistics is small or the p-value is high (greater than the significance level 5%)" \
                     f", then we cannot reject the hypothesis. But we got p={p} and s={s}"


def check_class_size(pop,
                     expected_class_size,
                     average_student_teacher_ratio,
                     average_student_all_staff_ratio,
                     err_margin=1):
    contact_nums = []
    for p in pop.values():
        if p["sc_student"] is not None:
            people = len(p["contacts"]["S"])
            staff = people // average_student_all_staff_ratio
            teacher = people // average_student_teacher_ratio
            contact_nums.append(people + 1 - staff - teacher)

    actual_class_size = np.mean(np.array(contact_nums))
    print(f"average class size:{actual_class_size}")
    assert expected_class_size - err_margin <= actual_class_size <= expected_class_size + err_margin, \
        f"expected class size: {expected_class_size} but actual class size: {actual_class_size}"

def get_average_contact_by_age(pop, datadir, state_location="Washington", country_location="usa", code="H", decimal=3):
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    total = np.zeros(len(brackets))
    contacts = np.zeros(len(brackets))
    for p in pop.values():
        total[ageindex[p["age"]]] += 1
        contacts[ageindex[p["age"]]] += len(p["contacts"][code])
    average = np.round(np.divide(contacts, total), decimals=decimal)
    return average

def rebin_matrix_by_age(matrix, datadir, state_location="Washington", country_location="usa"):
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    agg_matrix = sp.get_aggregate_matrix(matrix, ageindex)
    from collections import Counter
    counter = Counter(ageindex.values())
    for i in range(0, len(counter)):
        for j in range(0, len(counter)):
            agg_matrix[i,j] /= (counter[i] * counter[j])
    return agg_matrix

def resize_2darray(array, merge_size):
    """
    shrink the array by merging elements
    calculate average per (merge_size * merge_size area)
    """
    M = array.shape[0] // merge_size if array.shape[0] % merge_size == 0 else (array.shape[0] // merge_size + 1)
    N = array.shape[1] // merge_size if array.shape[1] % merge_size == 0 else (array.shape[1] // merge_size + 1)
    new_array = np.zeros((M,N))
    for i in range(0, M):
        x1 = i * merge_size
        x2 = (i+1) * merge_size if (i + 1) * merge_size < array.shape[0] else array.shape[0]
        for j in range(0, N):
            y1 = j * merge_size
            y2 = (j+1) * merge_size if (j + 1) * merge_size < array.shape[1] else array.shape[1]
            new_array[i,j] = np.nanmean(array[x1:x2, y1:y2])
    return new_array