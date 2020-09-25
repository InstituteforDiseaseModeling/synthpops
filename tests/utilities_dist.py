import synthpops as sp
import numpy as np
import utilities
def check_work_size_dist(pop,
                           n,
                           datadir,
                           location=None,
                           state_location=None,
                           country_location=None,
                           file_path=None,
                           use_default=False,
                           test_prefix="test",
                           skip_stat_check=False,
                           do_close=True):
    wb = sp.get_workplace_size_brackets(datadir, location, state_location, country_location, file_path, use_default)
    ws = sp.norm_dic(
        sp.get_workplace_size_distr_by_brackets(datadir, location, state_location, country_location, file_path, use_default)
    )
    ws_index = get_worksize_by_brackets_dic(wb)
    upper_bound = max(ws_index.keys())
    actual_work_dist = {}
    for p in pop.values():
        if p['wpid'] in actual_work_dist:
            actual_work_dist[p['wpid']] += 1
        else:
            actual_work_dist[p['wpid']]=1
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
    utilities.plot_array(expected_values, actual_values, names=wb.keys(), testprefix="work size distribution", do_close=False)
    if not skip_stat_check:
        utilities.statistic_test(expected_values, actual_values, test="x", comments="work size distribution check")

def get_worksize_by_brackets_dic(work_brackets):
    worksize_by_brackets_dic = {}
    for b in work_brackets:
        for a in work_brackets[b]:
            worksize_by_brackets_dic[a] = b
    return worksize_by_brackets_dic

def check_employment_age_dist(pop):
    pass

def check_household_dist(pop):
    pass