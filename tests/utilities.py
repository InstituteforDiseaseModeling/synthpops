import synthpops as sp
import sciris as sc
import inspect
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from scipy import stats


def runpop(resultdir, actual_vals, testprefix, method):

    """
    run any method which create apopulation
    and write args and population to file "{resultdir}/{testprefix}.txt"
    method must be a method which returns population
    and write population file to "{resultdir}/{testprefix}_pop.json"

    args:
      resultdir (str): result folder
      actual_vals (dict): a dictionary with param name and param value
      testprefix (str): test prefix to generate file name
    """
    os.makedirs(resultdir, exist_ok=True)
    params = {}
    args = inspect.getfullargspec(method).args
    for i in range(0, len(args)):
        params[args[i]] = inspect.signature(method).parameters[args[i]].default
    for name in actual_vals:
        if name in params.keys():
            params[name] = actual_vals[name]
    with open(os.path.join(resultdir, f"{testprefix}.txt"), mode="w") as cf:
        for key, value in params.items():
            cf.writelines(str(key) + ':' + str(value) + "\n")

    pop = method(**params)
    sc.savejson(os.path.join(resultdir, f"{testprefix}_pop.json"), pop, indent=2)
    return pop


def copy_input(sourcedir, resultdir, subdir_level):

    """
    Copy files to the target datadir up to the subdir level
    """

    # copy all files to datadir except the ignored files
    ignorepatterns = shutil.ignore_patterns("*contact_networks*",
                                            "*contact_networks_facilities*",
                                            "*New_York*",
                                            "*Oregon*")
    shutil.copytree(sourcedir, os.path.join(resultdir, subdir_level), ignore=ignorepatterns)


def check_teacher_staff_ratio(pop, datadir, test_prefix, average_student_teacher_ratio, average_student_all_staff_ratio, err_margin=0):

    """
    check if generated population matches
    average_student_teacher_ratio and average_student_all_staff_ratio

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
    assert(len(zero_teacher_case) == 0), \
        f"All schools with more students than the student teacher ratio should have at least one teacher. {len(zero_teacher_case)} did not."
    zero_staff_case = result.query('staff == 0 & student > @average_student_all_staff_ratio')
    assert(len(zero_staff_case) == 0), \
        f"All schools with more students than the student staff ratio: {average_student_all_staff_ratio} should have at least 1 staff. {len(zero_staff_case)} did not."

    # exclude 0 teacher if size is too small
    result = result[result.teacher > 0][result.staff > 0]

    # exclude student size less than 3*average_student_all_staff_ratio
    # average across school must match input
    actual_teacher_ratio = result[result["student"] > 3*average_student_teacher_ratio]["teacher_ratio"].mean()
    print(f"actual average student teacher ratio (ignore small size schools):{actual_teacher_ratio}")
    assert (int(average_student_teacher_ratio + err_margin) >= actual_teacher_ratio >= int(average_student_teacher_ratio - err_margin)), \
        f"teacher ratio: expected: {average_student_teacher_ratio} actual: {actual_teacher_ratio}"
    actual_staff_ratio = result[result["student"] > 3*average_student_teacher_ratio]["allstaff_ratio"].mean()
    print(f"actual average student all staff ratio (ignore small size schools):{actual_staff_ratio}")
    assert (int(average_student_all_staff_ratio + err_margin) >= actual_staff_ratio >= int(average_student_all_staff_ratio - err_margin)), \
        f"all staff ratio expected: {average_student_all_staff_ratio} actual: {actual_staff_ratio}"
    return result

def plot_array(expected, actual, names=None, datadir=None, testprefix="test"):
    """
    plot histogram on sorted array based by names
    if names not provided the order will be used
    """
    fig, ax = plt.subplots(1,1)
    font = {'weight': 'bold',
            'size': 10}
    plt.rc('font', **font)
    plt.title(f"Comparison for {testprefix}")

    names = range(0, len(expected)) if names is None else names
    ax.hist(x=names, alpha=0.5, weights=expected, label='expected',bins=len(expected))
    ax.hist(x=names, alpha=0.5, weights=actual, label='actual', bins=len(actual))
    ax.legend(loc='upper right')
    plt.show()
    if datadir:
        plt.savefig(os.path.join(os.path.dirname(datadir), f"{testprefix}_graph.png"), format="png")
    plt.close()

def check_age_distribution(pop,
                           n,
                           datadir,
                           location=None,
                           state_location=None,
                           country_location=None,
                           file_path=None,
                           use_default=False,
                           test_prefix="test", skip_stat_check=False):

    """
    construct histogram from expected age distribution
    compare with the actual generated data
    """
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    actual_age_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    for p in pop.values():
        for b in brackets:
            if p["age"] in brackets[b]:
                actual_age_dist[b] += 1
                break
    # un-normalized data
    #expected_values = np.array(list(age_dist.values())) * n
    #actual_values = np.array(list(actual_age_dist.values()))
    # normalized
    expected_values = np.array(list(age_dist.values()))
    actual_values = np.array(list(sp.norm_dic(actual_age_dist).values()))
    names = np.array(list(age_dist.keys()))
    plot_array(expected_values, actual_values, names, datadir, test_prefix + "_age")
    if not skip_stat_check:
        statistic_test(expected_values, actual_values, test="x", comments="age distribution check")


def check_enrollment_distribution(pop,
                                  n,
                                  datadir,
                                  location=None,
                                  state_location=None,
                                  country_location=None,
                                  file_path=None,
                                  use_default=False,
                                  test_prefix="test",
                                  skip_stat_check=False):

    """
    Compute the statistic on expected enrollment-age distribution and compare with actual distribution
    check zero enrollment bins to make sure there is nothing generated
    """
    expected_dist = sp.get_school_enrollment_rates(datadir, location, state_location, country_location, file_path, use_default)
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)

    # get actual school enrollment by age
    actual_pool = []
    actual_dist = dict.fromkeys(list(range(0, 101)), 0)
    for p in pop.values():
        if p["scid"] is not None and p["sc_student"] is not None:
            actual_dist[p["age"]] += 1
            actual_pool.append(p["age"])

    #adjust expected enrollment percentage by age brackets
    expected_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    actual_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    scaled_dist = dict.fromkeys(list(range(0, 101)), 0)
    for i in age_dist:
        for j in brackets[i]:
            scaled_dist[j] = (expected_dist[j] * n * age_dist[i])/len(brackets[i])
            expected_combined_dist[i] += scaled_dist[j]
            actual_combined_dist[i] += actual_dist[j]

    #construct expected pool based on adjusted distribution
    expected_pool =[]
    for key in scaled_dist:
        for i in range(0,int(scaled_dist[key])):
            expected_pool.append(key)

    print(f"total enroll expected :{int(sum(scaled_dist.values()))}")
    print(f"total enroll actual :{sum(actual_dist.values())}")

    #make sure results are sorted by key
    scaled_dist_dist = dict(sorted(scaled_dist.items()))
    actual_dist = dict(sorted(actual_dist.items()))

    expected_values = np.array(list(scaled_dist.values()))
    actual_values =  np.array(list(actual_dist.values()))
    expected_combined_values = np.array(list(expected_combined_dist.values()))
    actual_combined_values = np.array(list(actual_combined_dist.values()))

    #uncomment below if you need to plot and check data
    plot_array(expected_values, actual_values, None, datadir, test_prefix)
    plot_array(expected_combined_values, actual_combined_values, None, datadir, test_prefix + "_combined")
    np.savetxt(os.path.join(os.path.dirname(datadir), f"{test_prefix}_expected.csv"), expected_values, delimiter=",")
    np.savetxt(os.path.join(os.path.dirname(datadir), f"{test_prefix}_actual.csv"), actual_values, delimiter=",")

    # check for expected 0 count bins
    # if expected enrollment is 0, actual enrollment must be 0
    threshold = 9
    assert np.sum(actual_values[expected_values == 0]) ==0, \
        f"expected enrollment should be 0 for these age bins: " \
        f"{str(np.where((expected_values == 0) & (actual_values!=0)))}"

    # if expected is greater than some threshold, actual should not be 0
    assert len(actual_values[np.where((expected_values > threshold) & (actual_values == 0))]) == 0, \
        f"actual enrollment should not be 0 for these age bins: " \
        f"{str(np.where((expected_values > threshold) & (actual_values == 0)))}"

    # if bin count less than threshold use range check to allow up
    # not exceeding up to 2*threshold
    i = np.where((expected_values <= threshold) & (expected_values > 0))
    u = expected_values[i] + threshold #upper bound
    l = np.zeros(len(expected_values[i])) #lower bound can be 0
    assert (sum(l <= actual_values[i]) == len(actual_values[i]) and sum(actual_values[i] <= u) == len(actual_values[i])),\
    f"results show too much difference:\n" \
    f"expected:{expected_values[i]} \n actual:{actual_values[i]} \n" \
    f"please check these age bins: {i}"

    #check if pool looks right
    #h, bins = np.histogram(np.array(expected_pool), bins=100)
    #h, bins = np.histogram(np.array(actual_pool), bins=100)
    #plt.bar(bins[:-1],h,width=1)
    #plt.show()

    if not skip_stat_check:
        statistic_test(expected_pool, actual_pool, test="ks", comments="enrollment distribution check")
    # todo: theoretically this should work, however does not pass in our example
    #statistic_test(actual_combined_values[expected_combined_values > 0],
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
                     test_prefix="test",
                     err_margin =1):
    contact_nums =[ ]
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