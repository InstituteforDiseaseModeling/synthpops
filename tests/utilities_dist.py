import synthpops as sp
import numpy as np
import utilities
import os
from collections import Counter


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
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    wb = sp.get_workplace_size_brackets(datadir, location, state_location, country_location, file_path, use_default)
    ws = sp.norm_dic(
        sp.get_workplace_size_distr_by_brackets
        (datadir, location, state_location, country_location, file_path, use_default)
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
        # if ws_index[v] in actual_worksizes:
        #     actual_worksizes[ws_index[v]] += 1
        # else:
        #     actual_worksizes[ws_index[v]] = 1
    actual_values = np.zeros(len(ws.keys()))
    for i in range(0, len(ws.keys())):
        if i in actual_worksizes:
            actual_values[i] = actual_worksizes[i]
    actual_values = actual_values / np.nansum(actual_values)
    expected_values = np.array(list(ws.values()))
    xlabels = [str(wb[b][0]) + '-' + str(wb[b][-1]) for b in sorted(wb.keys())]
    utilities.plot_array(expected_values, actual_values, names=list(wb.keys()), datadir=figdir,
                         testprefix="work size distribution "+test_prefix, do_close=do_close,
                         xlabels=xlabels, xlabel_rotation=50)
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
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    er = sp.get_employment_rates(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    actual_employed_age_dist, actual_unemployed_age_dist = utilities.get_ids_count_by_param(pop, 'wpid', 'age')

    sorted_actual_employed_rate = {}
    actual_employed_rate = utilities.calc_rate(actual_employed_age_dist, actual_unemployed_age_dist)
    for i in er.keys():
        # sorted_actual_employed_rate.setdefault(i, 0)
        # sorted_actual_employed_rate[i] = actual_employed_rate[i]
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
    utilities.plot_array(expected_values, actual_values, names=names, datadir=figdir,
                         testprefix="employment rate distribution " + test_prefix, do_close=do_close)

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
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    hs = sp.get_household_size_distr(datadir, location=location,
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
    utilities.plot_array(expected_values, actual_values, names=hs.keys(), datadir=figdir,
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
                                   skip_stat_check=False,
                                   do_close=True):
    """
    Check the school size distribution against the reference data
    Args:
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    sb = sp.get_school_size_brackets(datadir,
                                     location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     file_path=file_path, use_default=use_default)
    sb_index = sp.get_index_by_brackets_dic(sb)

    sdf = sp.get_school_sizes_df(datadir, location=location,
                                 state_location=state_location,
                                 country_location=country_location)
    expected_scount = Counter(sdf.iloc[:, 0].values)
    expected_school_size_by_brackets = sp.norm_dic(utilities.get_bucket_count(sb_index, sb, expected_scount))
    actual_school, actual_school_none = utilities.get_ids_count_by_param(pop, "scid")
    actual_scount = dict(Counter(actual_school.values()))
    actual_school_size_by_brackets = sp.norm_dic(utilities.get_bucket_count(sb_index, sb, actual_scount))
    expected_values = np.array(list(expected_school_size_by_brackets.values()))
    actual_values = np.array(list(actual_school_size_by_brackets.values()))
    utilities.plot_array(expected_values, actual_values, names=sb.keys(), datadir=figdir,
                         testprefix="school size " + test_prefix, do_close=do_close)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="school size check")
    # check average school size
    expected_average_school_size = sum(sdf.iloc[:, 0].values) / len(sdf)
    actual_average_school_size = sum([i * actual_scount[i] for i in actual_scount]) / sum(actual_scount.values())
    utilities.check_error_percentage(n, expected_average_school_size, actual_average_school_size,
                                     name="average school size")


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
    Args:
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True
        plot_only: plot only without doing any data checks

    Returns:
        None
        Plots will be save to figdir if provided
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
    construct histogram from expected age distribution
    compare with the actual generated data
    Args:
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        location: location of the reference data
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        skip_stat_check: skip the statistics check for distribution
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
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
        pop: population dictionary
        n: population size
        datadir: root data directory which has resides the reference data
        figdir: directory where to result files are saved
        state_location: state location of the reference data
        country_location: country location of the reference data
        file_path: reference data path if specified, otherwise will be inferred from locations provided or use default
        use_default: use default location if set to True
        test_prefix: used for prefix of the plot title
        do_close: close the image immediately if set to True

    Returns:
        None
        Plots will be save to figdir if provided
    """
    df = sp.get_household_head_age_by_size_df(datadir,
                                              state_location=state_location,
                                              country_location=country_location,
                                              file_path=file_path,
                                              use_default=use_default)
    hh_index = utilities.get_household_age_brackets_index(df)
    expected_hh_ages = df.loc[:, df.columns.str.startswith("household_head_age")]
    expected_hh_ages_percentage = expected_hh_ages.div(expected_hh_ages.sum(axis=0), axis=1)
    actual_hh_ages_percetnage = utilities.get_household_head_age_size(pop, index=hh_index)
    expected_values = expected_hh_ages_percentage.values
    actual_values = actual_hh_ages_percetnage.values
    label_columns = df.columns[df.columns.str.startswith("household_head_age")].values
    xlabels = [lc.strip('household_head_age').replace('_', '-') for lc in label_columns]
    family_sizes = [i+2 for i in range(0, len(expected_hh_ages_percentage))]
    # ylabels = family_sizes
    utilities.plot_heatmap(expected_values, actual_values,
                           xlabels, family_sizes, 'Head of Household Age', 'Household Size',
                           # expected_hh_ages_percentage.columns, # family_sizes,
                           testprefix="household_head_age_family_size " + test_prefix,
                           figdir=figdir, do_close=do_close)
