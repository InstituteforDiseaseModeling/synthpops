import synthpops as sp
import numpy as np
import utilities
from collections import Counter


def check_work_size_dist(pop,
                           n,
                           datadir,
                           location=None,
                           state_location=None,
                           country_location=None,
                           file_path=None,
                           use_default=False,
                           test_prefix="",
                           skip_stat_check=False,
                           do_close=True):
    wb = sp.get_workplace_size_brackets(datadir, location, state_location, country_location, file_path, use_default)
    ws = sp.norm_dic(
        sp.get_workplace_size_distr_by_brackets(datadir, location, state_location, country_location, file_path, use_default)
    )
    ws_index = get_index_by_brackets_dic(wb)
    upper_bound = max(ws_index.keys())
    actual_work_dist, actual_work_dist_none = get_ids_count_by_param(pop, "wpid")
    actual_worksizes = {}
    for v in actual_work_dist.values():
        if v > upper_bound:
            v = upper_bound
        if ws_index[v] in actual_worksizes:
            actual_worksizes[ws_index[v]] +=1
        else:
            actual_worksizes[ws_index[v]] = 1
    actual_values = np.zeros(len(ws.keys()))
    for i in range(0, len(ws.keys())):
        if i in actual_worksizes:
            actual_values[i] = actual_worksizes[i]
    actual_values = actual_values / np.nansum(actual_values)
    expected_values = np.array(list(ws.values()))
    utilities.plot_array(expected_values, actual_values, names=wb.keys(),
                         testprefix="work size distribution "+test_prefix, do_close=do_close)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x", comments="work size distribution check")


def check_employment_age_dist(pop,
                              n,
                              datadir,
                              location=None,
                              state_location=None,
                              country_location=None,
                              file_path=None,
                              use_default=False,
                              test_prefix="",
                              skip_stat_check=False,
                              do_close=True):
    er = sp.get_employment_rates(datadir, location, state_location, country_location, file_path, use_default)
    brackets = sp.get_census_age_brackets(datadir, state_location, country_location)
    ageindex = sp.get_age_by_brackets_dic(brackets)
    age_dist = sp.read_age_bracket_distr(datadir, location, state_location, country_location, file_path, use_default)
    actual_employed_age_dist, actual_unemployed_age_dist = get_ids_count_by_param(pop, 'wpid', 'age')

    sorted_actual_employed_rate = {}
    actual_employed_rate = calc_rate(actual_employed_age_dist, actual_unemployed_age_dist)
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
    #plotting fill 0 to under age 16 for better display
    filled_count = min(er.keys())
    expected_values = np.insert(expected_values, 0, np.zeros(filled_count))
    actual_values = np.insert(actual_values, 0, np.zeros(filled_count))
    names = [i for i in range(0, max(er.keys())+1)]
    utilities.plot_array(expected_values, actual_values, names=names,
                         testprefix="employment rate distribution " + test_prefix, do_close=do_close)

    #check if total employment match
    expected_employed_brackets = {k: 0 for k in brackets}
    actual_employed_brackets ={k: 0 for k in brackets}
    for i in names:
        expected_employed_brackets[ageindex[i]] += expected_values[i]
        if i in actual_employed_age_dist:
            actual_employed_brackets[ageindex[i]] += actual_employed_age_dist[i]
    for i in expected_employed_brackets:
        expected_employed_brackets[i] = expected_employed_brackets[i] / len(brackets[i]) * age_dist[i] * n

    expected_total = np.array(list(expected_employed_brackets.values()))
    actual_total = np.array(list(actual_employed_brackets.values()))
    utilities.plot_array(expected_total, actual_total, names=brackets.keys(),
                         testprefix="employment total " + test_prefix, do_close=do_close)
    expected_etotal = np.round(np.sum(expected_total))
    actual_etotal = np.round(np.sum(actual_total))
    check_error_percentage(n, expected_etotal, actual_etotal, name="employee")


def check_household_dist(pop,
                         n,
                         datadir,
                         location=None,
                         state_location=None,
                         country_location=None,
                         file_path=None,
                         use_default=False,
                         test_prefix="",
                         skip_stat_check=False,
                         do_close=True):
    hs = sp.get_household_size_distr(datadir, location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     file_path=file_path,
                                     use_default=use_default)

    actual_households, actual_households_none = get_ids_count_by_param(pop, "hhid")
    assert actual_households_none == {}, "all entries must have household ids"
    actual_household_count = dict(Counter(actual_households.values()))
    sorted_actual_household_count = {}
    for i in sorted(actual_household_count):
        sorted_actual_household_count[i] = actual_household_count[i]
    actual_values = np.array(list(sp.norm_dic(sorted_actual_household_count).values()))
    expected_values = np.array(list(hs.values()))
    utilities.plot_array(expected_values, actual_values, names=hs.keys(),
                         testprefix="household count percentage " + test_prefix, do_close=do_close)

    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="household count percentage check")
    #check average household size
    expected_average_household_size = round(sum([(i+1)*expected_values[np.where(i)] for i in expected_values])[0], 3)
    actual_average_household_size = round(sum([(i+1)*actual_values[np.where(i)] for i in actual_values])[0], 3)
    print(f"expected average household size: {expected_average_household_size}\n"
          f"actual average household size: {actual_average_household_size}")
    check_error_percentage(n, expected_average_household_size, actual_average_household_size, name="average household size")


def check_school_size_dist(pop,
                         n,
                         datadir,
                         location=None,
                         state_location=None,
                         country_location=None,
                         file_path=None,
                         use_default=False,
                         test_prefix="",
                         skip_stat_check=False,
                         do_close=True):

    sb = sp.get_school_size_brackets(datadir,
                                     location=location,
                                     state_location=state_location,
                                     country_location=country_location,
                                     file_path=file_path, use_default=use_default)
    sb_index = get_index_by_brackets_dic(sb)

    sdf = sp.get_school_sizes_df(datadir, location=location,
                                 state_location=state_location,
                                 country_location=country_location)
    expected_scount = Counter(sdf.iloc[:, 0].values)
    expected_school_size_by_brackets = sp.norm_dic(get_bucket_count(sb_index, sb, expected_scount))
    actual_school, actual_school_none = get_ids_count_by_param(pop, "scid")
    actual_scount = dict(Counter(actual_school.values()))
    actual_school_size_by_brackets = sp.norm_dic(get_bucket_count(sb_index, sb, actual_scount))
    expected_values = np.array(list(expected_school_size_by_brackets.values()))
    actual_values = np.array(list(actual_school_size_by_brackets.values()))
    utilities.plot_array(expected_values, actual_values, names=sb.keys(),
                         testprefix="school size " + test_prefix, do_close=do_close)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x",
                                 comments="school size check")
    #check average school size
    expected_average_school_size = sum(sdf.iloc[:, 0].values) / len(sdf)
    actual_average_school_size = sum([i * actual_scount[i] for i in actual_scount]) / sum(actual_scount.values())
    check_error_percentage(n, expected_average_school_size, actual_average_school_size, name="average school size")

def get_index_by_brackets_dic(brackets):
    by_brackets_dic = {}
    for b in brackets:
        for a in brackets[b]:
            by_brackets_dic[a] = b
    return by_brackets_dic


def calc_rate(a, b):
    rate = dict()
    for k, v in a.items():
        rate[k] = v/(v + b[k])
    return rate

def sort_dict(d):
    new_dict = {}
    for i in d:
        new_dict[i] = d[i]
    return new_dict


def get_ids_count_by_param(pop, idname, param=None):
    ret = {}
    ret_none = {}
    param = idname if param is None else param
    for p in pop.values():
        if p[idname] is None:
            if p[param] in ret_none:
                ret_none[p[param]] += 1
            else:
                ret_none[p[param]] = 1
        else:
            if p[param] in ret:
                ret[p[param]] += 1
            else:
                ret[p[param]] = 1
    return ret, ret_none


def get_bucket_count(index, brackets, values):
    values_by_brackets = {k: 0 for k in brackets.keys()}
    for i in values:
        if i < min(index):
            values_by_brackets[index[min(index)]] += values[i]
        else:
            values_by_brackets[index[i]] += values[i]
    return values_by_brackets


def check_error_percentage(n, expected, actual, err_margin_percent=10, name="", assertfail=False):
    print(f"\nexpected {name} {expected}\n actual {name} {actual} \n for n={n}")
    # check if within err_margin_percent% error
    err = abs(actual - expected) / expected * 100.0
    print(f"percentage error: {np.round(err, 2)}%")
    if assertfail:
        assert err < err_margin_percent, f"failed with {err_margin_percent}% percentage error margin"
