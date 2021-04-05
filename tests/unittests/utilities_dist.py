import synthpops as sp
import numpy as np
import utilities
import os
from collections import Counter
import pandas

def check_work_size_distribution(pop,
                                 n,
                                 datadir,
                                 figdir,
                                 location=None,
                                 state_location=None,
                                 country_location=None,
                                 file_path=None,
                                 use_default=False,
                                 test_prefix="",
                                 skip_stat_check=False,
                                 do_close=True):
    """
    Check the population workplace size distribution against the reference data

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state the location is in
        country_location : name of the country the location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "work_size")
    wb = sp.get_workplace_size_brackets(datadir=datadir,
                                        location=location,
                                        state_location=state_location,
                                        country_location=country_location,
                                        file_path=file_path,
                                        use_default=use_default)
    ws = sp.norm_dic(
        sp.get_workplace_size_distr_by_brackets
        (datadir=datadir,
         location=location,
         state_location=state_location,
         country_location=country_location,
         file_path=file_path,
         use_default=use_default)
    )
    ws_index = sp.get_index_by_brackets_dic(wb)
    upper_bound = max(ws_index.keys())
    actual_work_dist, actual_work_dist_none = utilities.get_ids_count_by_param(pop, "wpid")
    actual_worksizes = {}
    for v in actual_work_dist.values():
        if v > upper_bound:
            v = upper_bound
        actual_worksizes.setdefault(ws_index[v], 0)
        actual_worksizes[ws_index[v]] += 1

    actual_values = np.zeros(len(ws.keys()))
    for i in range(0, len(ws.keys())):
        if i in actual_worksizes:
            actual_values[i] = actual_worksizes[i]
    actual_values = actual_values / np.nansum(actual_values)
    expected_values = np.array(list(ws.values()))
    xlabels = [str(wb[b][0]) + '-' + str(wb[b][-1]) for b in sorted(wb.keys())]
    utilities.plot_array(expected_values, actual_values, names=xlabels, datadir=figdir,
                         testprefix="work size distribution "+test_prefix, do_close=do_close, xlabel_rotation=50)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x", comments="work size distribution check")


def check_employment_age_distribution(pop,
                                      n,
                                      datadir,
                                      figdir,
                                      location=None,
                                      state_location=None,
                                      country_location=None,
                                      file_path=None,
                                      use_default=False,
                                      test_prefix="",
                                      skip_stat_check=False,
                                      do_close=True):
    """
    Check the population employment by age distribution against the reference data

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state the location is in
        country_location : name of the country the location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "employment")
    er = sp.get_employment_rates(datadir=datadir,
                                 location=location,
                                 state_location=state_location,
                                 country_location=country_location,
                                 file_path=file_path, use_default=use_default)
    brackets = sp.get_census_age_brackets(datadir=datadir,
                                          state_location=state_location,
                                          country_location=country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    age_dist = sp.read_age_bracket_distr(datadir=datadir,
                                         location=location,
                                         state_location=state_location,
                                         country_location=country_location,
                                         file_path=file_path,
                                         use_default=use_default)
    # counting the actual population by age with employment including teachers and staffs
    actual_employed_age_dist, actual_unemployed_age_dist = \
        utilities.get_ids_count_by_param(pop,
                                         condition_name=['wpid', 'sc_teacher', 'sc_staff'],
                                         param='age')
    utilities.plot_array([actual_employed_age_dist[k] for k in sorted(actual_employed_age_dist)],
                         datadir=figdir,
                         names =[k for k in sorted(actual_employed_age_dist)],
                         expect_label='employed by age count',
                         xlabel_rotation=90,
                         testprefix="employeed count by age " + test_prefix)
    utilities.plot_array([actual_unemployed_age_dist[k] for k in sorted(actual_unemployed_age_dist)],
                         datadir=figdir,
                         names=[k for k in sorted(actual_unemployed_age_dist)],
                         expect_label='unemployed by age count',
                         xlabel_rotation=90,
                         testprefix="unemployed count by age " + test_prefix)

    sorted_actual_employed_rate = {}
    actual_employed_rate = utilities.calc_rate(actual_employed_age_dist, actual_unemployed_age_dist)
    for i in er.keys():
        if i in actual_employed_rate:
            sorted_actual_employed_rate[i] = actual_employed_rate[i]
        else:
            sorted_actual_employed_rate[i] = 0
    actual_values = np.array(list(sorted_actual_employed_rate.values()))
    expected_values = np.array(list(er.values()))
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="employment rate distribution check")
    # plotting fill 0 to under age 16 for better display
    filled_count = min(er.keys())
    expected_values = np.insert(expected_values, 0, np.zeros(filled_count))
    actual_values = np.insert(actual_values, 0, np.zeros(filled_count))
    names = [i for i in range(0, max(er.keys())+1)]
    # somehow double stacks for age 100
    utilities.plot_array(expected_values, actual_values, names=None, datadir=figdir,
                         testprefix="employment rate distribution " + test_prefix, do_close=do_close, )

    # check if total employment match
    expected_employed_brackets = {k: 0 for k in brackets}
    actual_employed_brackets = {k: 0 for k in brackets}
    for i in names:
        expected_employed_brackets[ageindex[i]] += expected_values[i]
        if i in actual_employed_age_dist:
            actual_employed_brackets[ageindex[i]] += actual_employed_age_dist[i]
    for i in expected_employed_brackets:
        expected_employed_brackets[i] = expected_employed_brackets[i] / len(brackets[i]) * age_dist[i] * n

    expected_total = np.array(list(expected_employed_brackets.values()))
    actual_total = np.array(list(actual_employed_brackets.values()))
    utilities.plot_array(expected_total, actual_total, names=brackets.keys(), datadir=figdir,
                         testprefix="employment total " + test_prefix, do_close=do_close)
    expected_etotal = np.round(np.sum(expected_total))
    actual_etotal = np.round(np.sum(actual_total))
    utilities.check_error_percentage(n, expected_etotal, actual_etotal, name="employee")


def check_household_distribution(pop,
                                 n,
                                 datadir,
                                 figdir,
                                 location=None,
                                 state_location=None,
                                 country_location=None,
                                 file_path=None,
                                 use_default=False,
                                 test_prefix="",
                                 skip_stat_check=False,
                                 do_close=True):
    """
    Check the household size distribution against the reference data

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state the location is in
        country_location : name of the country the location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "household")
    hs = sp.get_household_size_distr(datadir=datadir, location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     file_path=file_path,
                                     use_default=use_default)
    actual_households, actual_households_none = utilities.get_ids_count_by_param(pop, "hhid")
    assert actual_households_none == {}, "all entries must have household ids"
    actual_household_count = dict(Counter(actual_households.values()))
    sorted_actual_household_count = {}
    for i in sorted(actual_household_count):
        sorted_actual_household_count[i] = actual_household_count[i]
    actual_values = np.array(list(sp.norm_dic(sorted_actual_household_count).values()))
    expected_values = np.array(list(hs.values()))
    utilities.plot_array(expected_values, actual_values, names=[x for x in list(hs.keys())], datadir=figdir,
                         testprefix="household count percentage " + test_prefix, do_close=do_close, value_text=True)

    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="household count percentage check")
    # check average household size
    expected_average_household_size = round(sum([(i+1)*expected_values[np.where(i)] for i in expected_values])[0], 3)
    actual_average_household_size = round(sum([(i+1)*actual_values[np.where(i)] for i in actual_values])[0], 3)
    print(f"expected average household size: {expected_average_household_size}\n"
          f"actual average household size: {actual_average_household_size}")
    utilities.check_error_percentage(n, expected_average_household_size, actual_average_household_size,
                                     name="average household size")


def check_school_size_distribution(pop,
                                   n,
                                   datadir,
                                   figdir,
                                   location=None,
                                   state_location=None,
                                   country_location=None,
                                   file_path=None,
                                   use_default=False,
                                   test_prefix="",
                                   skip_stat_check=True,
                                   do_close=True,
                                   school_type=None):
    """
    Check the school size distribution against the reference data

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state
        country_location : name of the country the state_location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True
        school_type      : list of school types e.g. ['pk', 'es', 'ms', 'hs', 'uv']

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "school_size")
    sb = sp.get_school_size_brackets(datadir=datadir,
                                     location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     file_path=file_path, use_default=use_default)
    sb_index = sp.get_index_by_brackets_dic(sb)



    expected_school_size_by_brackets = sp.get_school_size_distr_by_brackets(datadir=datadir,
                                                                            location=location,
                                                                            state_location=state_location,
                                                                            country_location=country_location)
    actual_school, actual_school_none = utilities.get_ids_count_by_param(pop, "scid")
    actual_school_student_only, actual_school_none_student_only = utilities.get_ids_count_by_param(pop, "sc_student", "scid")
    actual_per_school_type_dict = {}
    actual_per_school_type_dict_student_only = {}
    actual_per_school_type_dict["all"] = actual_school
    actual_per_school_type_dict_student_only["all"] = actual_school_student_only
    if school_type is not None:
        for sc in school_type:
            actual_per_school_type_dict[sc] = \
                utilities.get_ids_count_by_param(pop, "sc_type", param="scid", condition_value=sc)[0]
            actual_per_school_type_dict_student_only[sc] = \
                utilities.get_ids_count_by_param(pop, "sc_type", param="scid", condition_value=sc, filter_expression={'sc_student':'1'})[0]

    # get individual school type size distribution
    for k in actual_per_school_type_dict:
        actual_scount = dict(Counter(actual_per_school_type_dict[k].values()))
        actual_scount_student_only = dict(Counter(actual_per_school_type_dict_student_only[k].values()))
        actual_school_size_by_brackets = sp.norm_dic(utilities.get_bucket_count(sb_index, sb, actual_scount))
        expected_values = np.array(list(expected_school_size_by_brackets.values()))
        actual_values = np.array(list(actual_school_size_by_brackets.values()))
        utilities.plot_array(expected_values, actual_values, names=sb.keys(), datadir=figdir,
                   testprefix="school size " + test_prefix + " " + k,
                   do_close=do_close)
        utilities.plot_array(actual_per_school_type_dict[k].values(), datadir=figdir,
                             expect_label =f"school count: total {len(actual_per_school_type_dict[k])}",
                             testprefix="school size total\n" + test_prefix + " " + k, binned=False, do_close=do_close)
        utilities.plot_array(actual_per_school_type_dict_student_only[k].values(), datadir=figdir,
                             expect_label=f"school count: total {len(actual_per_school_type_dict[k])}",
                             testprefix="school size total (student only)\n" + test_prefix + " " + k,
                             binned=False, do_close=do_close)
        # statistic_test is not working yet because school sizes are now available by school type. Also depends strongly on population size.
        if not skip_stat_check:
            utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="school size check")
        # check average school size
        school_size_brackets = sp.get_school_size_brackets(datadir=datadir,
                                                           location=location,
                                                           country_location=country_location,
                                                           state_location=state_location)
        # calculate the average school size per bracket
        average_school_size_in_bracket = [sum(i)/len(i) for i in school_size_brackets.values()]

        # calculate expected school size based on expected value sum(distribution * size)
        expected_average_school_size = sum([v[1] * average_school_size_in_bracket[v[0]] for v in expected_school_size_by_brackets.items()])
        actual_average_school_size = sum([i * actual_scount[i] for i in actual_scount]) / sum(actual_scount.values())
        utilities.check_error_percentage(n, expected_average_school_size, actual_average_school_size,
                                     name=f"average school size:'{k}'")
    # check school count distribution
    utilities.plot_array([len(actual_per_school_type_dict[i]) for i in actual_per_school_type_dict],
                         names=list(actual_per_school_type_dict.keys()),
                         datadir=figdir,
                         expect_label="school count",
                         testprefix="school count " + test_prefix,
                         value_text=True)


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
                                  plot_only=False,
                                  school_type=None):
    """
    Compute the statistic on expected enrollment-age distribution and compare with actual distribution
    check zero enrollment bins to make sure there is nothing generated

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state
        country_location : name of the country the state_location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True
        plot_only        : plot only without doing any data checks
        school_type      : list of school types e.g. ['pk', 'es', 'ms', 'hs', 'uv']

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    expected_dist = sp.get_school_enrollment_rates(datadir=datadir,
                                                   location=location,
                                                   state_location=state_location,
                                                   country_location=country_location,
                                                   file_path=file_path,
                                                   use_default=use_default)
    age_dist = sp.read_age_bracket_distr(datadir=datadir,
                                         location=location,
                                         state_location=state_location,
                                         country_location=country_location,
                                         file_path=file_path,
                                         use_default=use_default)
    brackets = sp.get_census_age_brackets(datadir=datadir,
                                          state_location=state_location,
                                          country_location=country_location)

    figdir = os.path.join(figdir, "enrollment")
    # get actual school enrollment by age
    if school_type is not None:
        actual_per_school_type_dict = dict.fromkeys(school_type)
        for sc in school_type:
            actual_per_school_type_dict[sc] = dict.fromkeys(list(range(0, 101)), 0)
    else:
        actual_per_school_type_dict = {}
    actual_pool = []
    actual_dist = dict.fromkeys(list(range(0, 101)), 0)
    for p in pop.values():
            if p["scid"] is not None and p["sc_student"] is not None:
                for sc in actual_per_school_type_dict.keys():
                    if p["sc_type"] == sc:
                        actual_per_school_type_dict[sc][p["age"]] += 1
                actual_dist[p["age"]] += 1
                actual_pool.append(p["age"])

    # plot total school enrollment and individual age distribution
    actual_per_school_type_dict["all"]=actual_dist
    if school_type is not None:
        utilities.plot_array([sum(actual_per_school_type_dict[i].values()) for i in actual_per_school_type_dict.keys()],
                             names=actual_per_school_type_dict.keys(), datadir=figdir,
                             testprefix= "enrollment_by_school_type\n" + test_prefix,
                             expect_label ="enrollment", value_text=True, do_close=do_close)
    for k in actual_per_school_type_dict:
        utilities.plot_array(actual_per_school_type_dict[k].values(), datadir=figdir,
                             testprefix=f"enrollment_by_age {k}\n" + test_prefix,
                             expect_label="enrollment by age bucket", do_close=do_close)

    actual_age_dist = utilities.get_age_distribution_from_pop(pop, brackets)
    # adjust expected enrollment percentage by age brackets
    expected_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    adjusted_expected_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)
    actual_combined_dist = dict.fromkeys(list(range(0, len(brackets))), 0)

    scaled_dist = dict.fromkeys(list(range(0, 101)), 0)
    adjusted_scaled_dist = dict.fromkeys(list(range(0, 101)), 0)
    for i in age_dist:
        for j in brackets[i]:
            scaled_dist[j] = (expected_dist[j] * n * age_dist[i]) / len(brackets[i])
            adjusted_scaled_dist[j] = (expected_dist[j] * n * actual_age_dist[i]) / len(brackets[i])
            expected_combined_dist[i] += scaled_dist[j]
            adjusted_expected_combined_dist[i] += adjusted_scaled_dist[j]
            actual_combined_dist[i] += actual_dist[j]

    # construct expected pool adjusted based on expected age distribution
    expected_pool = []
    for key in scaled_dist:
        for i in range(0, int(scaled_dist[key])):
            expected_pool.append(key)

    # construct expected pool adjusted based on the actual age distribution
    adjusted_expected_pool = []
    for key in adjusted_scaled_dist:
        for i in range(0, int(adjusted_scaled_dist[key])):
            adjusted_expected_pool.append(key)

    print(f"total enrollment expected :{int(sum(scaled_dist.values()))}")
    print(f"total enrollment expected (adjusted) :{int(sum(adjusted_scaled_dist.values()))}")
    print(f"total enrollment actual :{sum(actual_dist.values())}")

    # make sure results are sorted by key
    # scaled_dist_dist = dict(sorted(scaled_dist.items()))
    actual_dist = dict(sorted(actual_dist.items()))

    expected_values = np.array(list(scaled_dist.values()))
    adjusted_expected_values = np.array(list(adjusted_scaled_dist.values()))
    actual_values = np.array(list(actual_dist.values()))

    expected_combined_values = np.array(list(expected_combined_dist.values()))
    adjusted_expected_combined_values = np.array(list(adjusted_expected_combined_dist.values()))
    actual_combined_values = np.array(list(actual_combined_dist.values()))

    utilities.plot_array(expected_values, actual_values, None, figdir,
                         "enrollment_" + test_prefix, do_close=do_close)
    utilities.plot_array(adjusted_expected_values, actual_values, None, figdir,
                         "adjusted enrollment_" + test_prefix, do_close=do_close)

    utilities.plot_array(expected_combined_values, actual_combined_values,
                         np.array([i[0] for i in brackets.values()]), figdir,
                         "enrollment by age bin" + test_prefix, do_close=do_close)
    utilities.plot_array(adjusted_expected_combined_values, actual_combined_values,
                         np.array([i[0] for i in brackets.values()]), figdir,
                         "adjusted enrollment by age bin" + test_prefix, do_close=do_close)
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

    # if expected bin count is less than threshold, use range check to allow some buffer
    # this is usually observed in smaller population in that expected count is small
    # so we allow actual observations to be 0 and up to the expected value plus threshold

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
        utilities.statistic_test(adjusted_expected_pool, actual_pool, test="ks",
                                 comments="enrollment distribution check")
    # todo: theoretically this should work, however does not pass in our example
    # statistic_test(actual_combined_values[expected_combined_values > 0],
    # expected_combined_values[expected_combined_values > 0], test="x")


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
    Construct histogram from expected age distribution and compare with the actual generated data.

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        location         : name of the location
        state_location   : name of the state the location is in
        country_location : name of the country the location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        skip_stat_check  : skip the statistics check for distribution
        do_close         : close the image immediately if set to True

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "age_distribution")
    age_dist = sp.read_age_bracket_distr(datadir=datadir,
                                         location=location,
                                         state_location=state_location,
                                         country_location=country_location,
                                         file_path =file_path,
                                         use_default=use_default)
    brackets = sp.get_census_age_brackets(datadir=datadir,
                                          state_location=state_location,
                                          country_location=country_location)
    # un-normalized data
    # expected_values = np.array(list(age_dist.values())) * n
    # actual_values = get_age_distribution_from_pop(pop, brackets, False)
    # normalized
    expected_values = np.array(list(age_dist.values()))
    actual_values = utilities.get_age_distribution_from_pop(pop, brackets)
    names = np.array([i[0] for i in brackets.values()])
    utilities.plot_array(expected_values, actual_values, names, figdir,
                         "age_distribution_" + test_prefix, do_close=do_close)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x", comments="age distribution check")


def check_household_head(pop,
                         n,
                         datadir,
                         figdir,
                         state_location=None,
                         country_location=None,
                         file_path=None,
                         use_default=False,
                         test_prefix="",
                         do_close=True):
    """
    Check the household head by age distribution against the reference data

    Args:
        pop              : population dictionary
        n                : population size
        datadir          : root data directory which has resides the reference data
        figdir           : directory where to result files are saved
        state_location   : name of the state the location is in
        country_location : name of the country the location is in
        file_path        : file path to user specified gender by age bracket distribution data
        use_default      : if True, try to first use the other parameters to find data specific to the location
                           under study, otherwise returns default data drawing from Seattle, Washington.
        test_prefix      : used for prefix of the plot title
        do_close         : close the image immediately if set to True

    Returns:
        None.

    Plots will be save to figdir if provided
    """
    figdir = os.path.join(figdir, "household_head")

    household_head_age_distribution_by_family_size = sp.get_head_age_by_size_distr(state_location=state_location,
                                                                                   country_location=country_location)
    head_age_brackets = sp.get_head_age_brackets(state_location=state_location,
                                        country_location=country_location)
    # Inverse the mapping for use below
    hha_index = sp.get_index_by_brackets_dic(head_age_brackets)

    household_head_age_distribution_by_family_size = household_head_age_distribution_by_family_size[1:]
    expected_hh_ages = pandas.DataFrame(household_head_age_distribution_by_family_size)
    expected_hh_ages_percentage = expected_hh_ages.div(expected_hh_ages.sum(axis=0), axis=1)
    actual_hh_ages_percetnage = utilities.get_household_head_age_size(pop, index=hha_index)
    expected_values = expected_hh_ages_percentage.values[1:,:]
    actual_values = actual_hh_ages_percetnage.values
    xlabels = [ f'{min(head_age_brackets[bracket_index])}-{max(head_age_brackets[bracket_index])}'
                for bracket_index in head_age_brackets.keys() ]
    family_sizes = [i+2 for i in range(0, len(expected_hh_ages_percentage))]
    utilities.plot_heatmap(expected_values, actual_values,
                           xlabels, family_sizes, 'Head of Household Age', 'Household Size',
                           # expected_hh_ages_percentage.columns, # family_sizes,
                           testprefix="household_head_age_family_size " + test_prefix,
                           figdir=figdir, do_close=do_close)
