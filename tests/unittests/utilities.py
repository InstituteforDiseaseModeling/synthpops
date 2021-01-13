"""
Utilities for test_school_staff (and possibly other functions)
"""


import sciris as sc
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import synthpops as sp
from synthpops import data_distributions as spdd
from synthpops import base as spb

# set the font family if available
mplt.rcParams['font.family'] = 'Roboto Condensed'

do_save = False


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
      method (str)       : method which generates population

    Returns:
        Population dictionary.
    """
    print('NB: calling runpop with "method" is not used, since only make_population() is used')
    os.makedirs(resultdir, exist_ok=True)
    params = {}
    for name in actual_vals:
        if name not in ['self', 'test_prefix', 'filename']: # Remove automatically generated but invalid parameters
            params[name] = actual_vals[name]
    sc.savejson(os.path.join(resultdir, f"{testprefix}.config.json"), params, indent=2)
    print(params)
    pop = sp.make_population(**params)
    if do_save:
        sc.savejson(os.path.join(resultdir, f"{testprefix}_pop.json"), pop, indent=2)
    return pop


def copy_input(sourcedir, resultdir, subdir_level, patterns=None):
    """
    Copy files to the target datadir up to the subdir level.

    Args:
        sourcedir (str)    : source directory
        resultdir (str)    : result directory
        subdir_level (str) : subdirectory

    Returns:
        None.
    """

    # copy all files to datadir except the ignored files
    patterns = ["*contact_networks*", "*contact_networks_facilities*", "*New_York*", "*Oregon*"] if patterns is None else patterns
    ignorepatterns = shutil.ignore_patterns(*patterns)
    shutil.copytree(sourcedir, os.path.join(resultdir, subdir_level), ignore=ignorepatterns)


def check_teacher_staff_ratio(pop, datadir, test_prefix, average_student_teacher_ratio, average_student_all_staff_ratio,
                              err_margin=0):
    """
    Check if generated population matches average_student_teacher_ratio and
    average_student_all_staff_ratio.

    Args:
        pop (dict)                              : population dictionary
        datadir (str)                           : data directory
        test_prefix (str)                       : prefix str for test
        average_student_teacher_ratio (float)   : The average number of students per teacher.
        average_student_all_staff_ratio (float) : The average number of students per staff members at school
                                                  (including both teachers and non teachers).
        err_margin (float)                      : error margin

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
        f"All schools with more students than the student teacher ratio " \
        f"should have at least one teacher. {len(zero_teacher_case)} did not."
    zero_staff_case = result.query('staff == 0 & student > @average_student_all_staff_ratio')
    assert (len(zero_staff_case) == 0), \
        f"All schools with more students than the student staff ratio: " \
        f"{average_student_all_staff_ratio} should have at least 1 staff. {len(zero_staff_case)} did not."

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


def plot_array(expected,
               actual=None,
               names=None,
               datadir=None,
               testprefix="test",
               do_close=True,
               expect_label='Expected',
               value_text=False,
               xlabels=None,
               xlabel_rotation=0,
               binned = True):
    """
    Plot histogram on a sorted array based by names. If names not provided the
    order will be used. If actual data is not provided, plot only the expected values.
    Note this can only be used with the limitation that data that has already been binned

    Args:
        expected        : Array of expected values
        actual          : Array of actual values
        names           : names to display on x-axis, default is set to the indexes of data
        datadir         : directory to save the plot if provided
        testprefix      : used to prefix the title of the plot
        do_close        : close the plot immediately if set to True
        expect_label    : Label to show in the plot, default to "expected"
        value_text      : display the values on top of the bar if specified
        xlabel_rotation : rotation degree for x labels if specified
        binned          : default to True, if False, it will just plot a simple histogram for expected data

    Returns:
        None.

    Plot will be saved in datadir if given
    """
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    font = {
            'size': 14
            }
    plt.rc('font', **font)
    mplt.rcParams['font.family'] = 'Roboto Condensed'
    title = testprefix if actual is None else f"Comparison for \n{testprefix}"
    ax.set_title(title)
    x = np.arange(len(expected))
    # if len(x) > 1:
    #     width = np.min(np.diff(x))/3   # not necessary
    # else:
    #     width = 1.
    if not binned:
        ax.hist(expected, label=expect_label.title(), color='skyblue')
    else:
        rect1 = ax.bar(x, expected, label=expect_label.title(), color='mediumseagreen')
        # ax.hist(x=names, histtype='bar', weights=expected, label=expect_label.title(), bins=bin, rwidth=1, color='#1a9ac0', align='left')
        if actual is not None:
            line, = ax.plot(x, actual, color='#3f75a2', marker='o', markersize=4, label='Actual')
            # arr = ax.hist(x=names, histtype='step', linewidth=3, weights=actual, label='Actual', bins=len(actual), rwidth=1, color='#ff957a', align='left')
        if value_text:
            autolabel(ax, rect1, 0, 5)
            if actual is not None:
                for j, v in enumerate(actual):
                    ax.text(j, v, str(round(v, 3)), fontsize=10, horizontalalignment='right', verticalalignment='top', color='#3f75a2')
        if names is not None:
            if isinstance(names, dict):
                xticks = sorted(names.keys())
                xticklabels = [names[k] for k in xticks]
            else:
                xticks = np.arange(len(names))
                xticklabels = names
            # plt.locator_params(axis='x', nbins=len(xticks))
            # if the labels are too many it will look crowded so we only show every 10 ticks
            if len(names) > 30:
                xticks = xticks[0::10]
                xticklabels = xticklabels[0::10]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=xlabel_rotation)
            # ax.set_xlim(left=-1)
    ax.legend(loc='upper right')

    if datadir:
        os.makedirs(datadir, exist_ok=True)
        plt.savefig(os.path.join(datadir, f"{testprefix}_graph.png".replace('\n', '_')), format="png")
    if do_close:
        plt.close()
    else:
        plt.show()
    return


def autolabel(ax, rects, h_offset=0, v_offset=0.3):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Args:
        ax       : matplotlib.axes figure object
        rects    : matplotlib.container.BarContainer
        h_offset : The position x to place the text at.
        v_offset : The position y to place the text at.

    Returns:
        None.

    Set the annotation according to the input parameters
    """
    for rect in rects:
        height = rect.get_height()
        text = ax.annotate('{}'.format(round(height, 3)),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(h_offset, v_offset),
                           textcoords="offset points",
                           ha='center', va='bottom')
        text.set_fontsize(10)


def statistic_test(expected, actual, test="ks", comments=""):
    """
    Perform statistics checks for exepected and actual data
    based on the null hypothesis that expected/actual distributions are identical
    throw assertion if the expected/actual differ significantly based on the test selected
    Args:
        expected : expected data (array)
        actual   : actual data (array)
        test     : "ks" for Kolmogorov-Smirnov, "x" for Chi-square statistic
        comments : for printing information only

    Returns:
        None.
    """
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
    """
    Check the average class to see if it matches the expected class size.
    Total Students were based on length of contacts mius teachers and staffs (calculated from
    average_student_teacher_ratio and average_student_all_staff_ratio
    Args:
        pop                             : population dictionary
        expected_class_size             : expected class size
        average_student_teacher_ratio   : average student teacher ratio
        average_student_all_staff_ratio : average student allstaff ratio
        err_margin                      : error margin allowed, default to 1

    Returns:
        None.
    """
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


def get_average_contact_by_age(pop, datadir, location="seattle_metro", state_location="Washington", country_location="usa", setting_code="H", decimal=3):
    """
    Helper method to get average contacts by age brackets
    Args:
        pop              : population dictionary
        datadir          : data directory to look up reference data
        state_location   : state location
        country_location : country location
        setting_code     : contact layer code, can be "H", "W", "S"
        decimal          : digits for rounding, default to 3

    Returns:
        numpy.ndarray: A numpy array with average contacts by age brackets.

    """
    brackets = spdd.get_census_age_brackets(datadir, location, state_location, country_location)
    ageindex = spb.get_age_by_brackets_dic(brackets)
    total = np.zeros(len(brackets))
    contacts = np.zeros(len(brackets))
    for p in pop.values():
        total[ageindex[p["age"]]] += 1
        contacts[ageindex[p["age"]]] += len(p["contacts"][setting_code])
    average = np.round(np.divide(contacts, total), decimals=decimal)
    return average


def rebin_matrix_by_age(matrix, datadir, location="seattle_metro", state_location="Washington", country_location="usa"):
    """
    Helper method to get the average of contact matrix by age brackets
    @TODO: should we merge the functionalities with sp.get_aggregate_matrix
    or remove as this operation may not be scientifically meaningful (?)

    Args:
        matrix           : raw matrix with single age bracket
        datadir          : data directory
        state_location   : state location
        country_location : country location

    Returns:
        numpy.ndarray: A matrix with desired age bracket with average values for all cells.

    """
    brackets = sp.get_census_age_brackets(datadir, location, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    agg_matrix = sp.get_aggregate_matrix(matrix, ageindex)
    counter = Counter(ageindex.values())  # number of ageindex per bracket
    for i in range(0, len(counter)):
        for j in range(0, len(counter)):
            agg_matrix[i, j] /= (counter[i] * counter[j])
    return agg_matrix


def resize_2darray(array, merge_size):
    """
    Helper method to shrink the array by merging elements. Calculates average per (merge_size * merge_size area).
    For example, an array of 100*100 with resize_2darray(array, 10) will yield 10*10 matrix.

    Args:
        array      : original array
        merge_size : how many cells to merge

    Returns:
        a reduced-size array
    """
    m = array.shape[0] // merge_size if array.shape[0] % merge_size == 0 else (array.shape[0] // merge_size + 1)
    n = array.shape[1] // merge_size if array.shape[1] % merge_size == 0 else (array.shape[1] // merge_size + 1)
    new_array = np.zeros((m, n))
    for i in range(0, m):
        x1 = i * merge_size
        x2 = (i+1) * merge_size if (i + 1) * merge_size < array.shape[0] else array.shape[0]
        for j in range(0, n):
            y1 = j * merge_size
            y2 = (j+1) * merge_size if (j + 1) * merge_size < array.shape[1] else array.shape[1]
            new_array[i, j] = np.nanmean(array[x1:x2, y1:y2])
    return new_array


def get_age_distribution_from_pop(pop, brackets, normalized=True):
    """
    Get age distribution from the population dictionary

    Args:
        pop        : population dictionary
        brackets   : age brackets
        normalized : weather the result is normalized, default to True

    Returns:
        a dictionary with age distribution by brackets
    """
    ageindex = sp.get_age_by_brackets_dic(brackets)
    actual_age_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    for p in pop.values():
        actual_age_dist[ageindex[p['age']]] += 1
    if normalized:
        actual_values = np.array(list(sp.norm_dic(actual_age_dist).values()))
    else:
        actual_values = np.array(list(actual_age_dist.values()))
    return actual_values


def calc_rate(a, b):
    """
    Helper method to calculate the rate r= a/(a+b) giving a, b are both dictionaries

    Args:
        a: a dictionary used as numerator
        b: a dictionary used as part of denominator (a+b)

    Returns:
        a new dictionary with calculated rate r= a/(a+b)
    """
    rate = dict()
    for k, v in a.items():
        rate[k] = v/(v + b[k])
    return rate


def sort_dict(d):
    """

    Helper method to sort the dictionary by it's key

    Args:
        d: input dictionary

    Returns:
        sorted dictionary by input dictionary's key
    """
    new_dict = {}
    for i in d:
        new_dict[i] = d[i]
    return new_dict


def get_ids_count_by_param(pop, condition_name, param=None, condition_value=None, filter_expression=None):
    """
    Helper method to count by parameter from the population dictionary
    for example get_ids_count_by_param(pop, "wpid", param="age") will count by age
    with respect to nodes with or without "wpid"

    Args:
        pop               : population dictionary
        condition_name    : the field to be used for filtering the target population, use param if not specified
        param             : the field to be used for counting
        condition_value   : the value to be used for filtering the condition_name, if set to None, all values will be considered
        filter_expression : dictionary represent filters used to further filter the data, e,g. {sc_type:'es'}

    Returns:
        ret      : a dictionary with count by param of which condition_name exists
        ret_none : a dictionary with count by param of which condition_name not exist
    """
    ret = {}
    ret_none = {}
    param = condition_name if param is None else param
    # allow condition_name/values to be list, note that condition should be exclusive
    # otherwise it may count the same person twice
    if type(condition_name) != list:
        condition_name = [condition_name]
    if condition_value and type(condition_value) != list:
        condition_value = [condition_value]
    for p in pop.values():
        matched = False
        for i in range(len(condition_name)):
            if p[condition_name[i]] is not None:
                matched =True
                if condition_value is None or (condition_value[i] is not None and p[condition_name[i]] == condition_value[i]):
                    skip = False
                    if filter_expression is not None:
                        for f in filter_expression:
                            if str(p[f]) != filter_expression[f]:
                                skip = True
                                break
                    if not skip:
                        ret.setdefault(p[param], 0)
                        ret[p[param]] += 1
        if not matched:
            ret_none.setdefault(p[param], 0)
            ret_none[p[param]] += 1
    return ret, ret_none


def get_bucket_count(index, brackets, values):
    """
    Helper method to sum the total count for each bucket,
    if the value exceeds the upper bound or falls below lower bound,
    it will be counted within the last or the first brackets respectively

    Args:
        index    : bracket index dictionary
        brackets : a dictionary with values as list of numbers (representing ranges)
        values   : a dictionary of which keys are the range numbers

    Returns:
        dict: A dictionary that the key is the index and the values are total count within that bucket index
    """
    values_by_brackets = {k: 0 for k in brackets.keys()}
    for i in values:
        if i < min(index):
            values_by_brackets[index[min(index)]] += values[i]
        elif i > max(index):
            values_by_brackets[index[max(index)]] += values[i]
        else:
            values_by_brackets[index[i]] += values[i]
    return values_by_brackets


def check_error_percentage(n, expected, actual, err_margin_percent=10, name="", assertfail=False):
    """

    Args:
        n                  : population size
        expected           : expected value (float)
        actual             : actual value (float)
        err_margin_percent : percentage of error margin between 0 to 100, default is 10 (10%)
        name               : name of the checked value, used for display only
        assertfail         : fail the tests if set to True, display info only if set to False

    Returns:
        None.
    """
    print(f"\nexpected {name} {expected}\n actual {name} {actual} \n for n={n}")
    # check if within err_margin_percent% error
    err = abs(actual - expected) / expected * 100.0
    print(f"percentage error: {np.round(err, 2)}%")
    if assertfail:
        assert err < err_margin_percent, f"failed with {err_margin_percent}% percentage error margin"


def get_household_head_age_size(pop, index):
    """
    Calculate the household count by household size and household head's age
    assuming lowest uids in the household should be household head

    Args:
        pop   : population dictionary
        index : household head's age bracket index

    Returns:
        dataframe with normalized household count by size and household head's age
    """
    df_household_age = pd.DataFrame(columns=['hhid', 'age', 'size', 'age_bracket'])
    for uid in sorted(pop):
        if pop[uid]["hhid"] not in df_household_age['hhid'].values:
            df_household_age = df_household_age.append({'hhid': pop[uid]['hhid'],
                                                        'age': pop[uid]['age'],
                                                        'size': len(pop[uid]['contacts']['H'])+1,
                                                        'age_bracket': index[pop[uid]['age']]},
                                                       ignore_index=True)

    df_household_age = df_household_age.groupby(['age_bracket', 'size'], as_index=False).count()\
        .pivot(index='size', columns='age_bracket', values='hhid').reset_index().drop(["size"], axis=1)
    df_household_age = df_household_age[[c for c in df_household_age.columns if type(c) is int]]
    df_household_age = df_household_age.div(df_household_age.sum(axis=0), axis=1)
    # ignore family size =1
    return df_household_age.drop(index=0)


def get_household_age_brackets_index(df):
    """
    Helper method to process data from sp.get_household_head_age_by_size_df

    Args:
        df: dataframe obtained from sp.get_household_head_age_by_size_df

    Returns:
        dict: A dictionary of household head's age brackets' index
    """
    age_dict = {}
    index = 0
    for c in df.columns:
        if c.startswith("household_head_age_"):
            age_range = str(c)[19:].split("_")
            for i in range(int(age_range[0]), int(age_range[1])+1):
                age_dict[i] = index
            index += 1
    return age_dict


def plot_heatmap(expected,
                 actual,
                 names_x,
                 names_y,
                 xlabel,
                 ylabel,
                 figdir=None,
                 testprefix="test",
                 do_close=True,
                 range=[0, 1]):
    """
    Plotting a heatmap of matrix

    Args:
        expected   : expected 2-dimenional matrix
        actual     : actual 2-dimenional matrix
        names_x    : name for x-axis
        names_y    : name for y-axis
        xlabel     :
        ylabel     :
        figdir     : directory where to result files are saved
        testprefix : used for prefix of the plot title
        do_close   : close the image immediately if set to True
        range      : range for heatmap's [vmin,vmax], default to [0,1]

    Returns:
        None.

        Plots will be save to figdir if provided
    """
    fig, axs = plt.subplots(1, 2, figsize=(17, 8),
                            # subplot_kw={'aspect': 1},
                            # gridspec_kw={'width_ratios': [1, 1]}
                            )
    fig.subplots_adjust(top=0.8, right=0.8, wspace=0.15)

    font = {'weight': 'bold',
            'size': 14}
    plt.rc('font', **font)
    im1 = axs[0].imshow(expected, origin='lower', cmap='viridis', interpolation='nearest', aspect="auto", vmin=range[0], vmax=range[1])
    im2 = axs[1].imshow(actual, origin='lower', cmap='viridis', interpolation='nearest', aspect="auto", vmin=range[0], vmax=range[1])
    for ax in axs:
        ax.set_xticks(np.arange(len(names_x)))
        ax.set_yticks(np.arange(len(names_y)))
        ax.set_xticklabels(names_x)
        ax.set_yticklabels(names_y)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    axs[0].set_title(f"Expected")
    axs[1].set_title(f"Actual")
    # plt.tight_layout()
    fig.suptitle(testprefix, fontsize=28)

    divider = make_axes_locatable(axs[1])
    cax = divider.new_horizontal(size='5%', pad=0.15)
    fig.add_axes(cax)
    cbar = fig.colorbar(im1, cax=cax)
    # cbar.ax.set_xlabel('')

    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im1, cax=cbar_ax)
    if figdir:
        os.makedirs(figdir, exist_ok=True)
        plt.savefig(os.path.join(figdir, f"{testprefix}_graph.png"), format="png", bbox_inches="tight")
    if do_close:
        plt.close()
    else:
        plt.show()
    return
