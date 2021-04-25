"""
Modeling Seattle Metro Long Term Care Facilities

"""

import numpy as np
import sciris as sc
from collections import Counter
from .config import logger as log, checkmem
from . import sampling as spsamp
from . import data_distributions as spdata
from . import base as spb


def generate_ltcfs(n, with_facilities, loc_pars, expected_age_dist, ages_left_to_assign):
    """
    Generate residents living in long term care facilities and their ages.

    Args:
        n (int)                   : The number of people to generate in the population
        with_facilities (bool)    : If True, create long term care facilities, currently only available for locations in the US.
        loc_pars (dict)           : A dictionary of location parameters
        expected_age_dist (dict)  : The expected age distribution
        ages_left_to_assign (dic) : The counter of ages for the generated population left to place in a residence
    """
    log.debug('generate_ltcfs()')
    # initialize an empty list for facilities
    facilities = []

    # If not using facilities, skip everything here
    if with_facilities:

        # what the ltcf user rates by age?
        ltcf_rates_by_age = spdata.get_long_term_care_facility_use_rates(loc_pars.datadir, country_location=loc_pars.country_location, state_location=loc_pars.state_location)

        # generate the count of ltcf users by age and make a list of all users represented by their age
        expected_users_by_age = dict.fromkeys(expected_age_dist.keys(), 0)

        # make a list of all resident ages
        all_residents = []
        for a in expected_users_by_age:
            expected_users_by_age[a] = np.random.binomial(ages_left_to_assign[a], ltcf_rates_by_age[a])
            all_residents.extend([a] * expected_users_by_age[a])

        # shuffle resident ages
        np.random.shuffle(all_residents)

        # how big are long term care facilities
        resident_size_dist = spb.norm_dic(spdata.get_long_term_care_facility_residents_distr(**loc_pars))
        resident_size_brackets = spdata.get_long_term_care_facility_residents_distr_brackets(**loc_pars)

        size_bracket_keys = sorted(resident_size_dist.keys())
        size_dist = [resident_size_dist[k] for k in size_bracket_keys]

        # create facilities
        while len(all_residents) > 0:

            b = spsamp.fast_choice(size_dist)
            size = np.random.choice(resident_size_brackets[b])

            if size > len(all_residents):
                size = len(all_residents)
            new_facility = all_residents[:size]
            facilities.append(new_facility)
            all_residents = all_residents[size:]

        # what's the age distribution and count of people left to place in a residence?
        ltcf_adjusted_age_dist = sc.dcp(expected_age_dist)
        for a in ltcf_adjusted_age_dist:
            ltcf_adjusted_age_dist[a] -= expected_users_by_age[a] / n
            ltcf_adjusted_age_dist[a] = max(ltcf_adjusted_age_dist[a], 0)
            ages_left_to_assign[a] -= expected_users_by_age[a]
        ltcf_adjusted_age_dist_values = np.array([ltcf_adjusted_age_dist[a] for a in ltcf_adjusted_age_dist.keys()])

        n_nonltcf = int(n - sum([len(facililty) for facililty in facilities]))

    else:
        n_nonltcf = n
        ltcf_adjusted_age_dist = sc.dcp(expected_age_dist)
        ltcf_adjusted_age_dist_values = np.array([ltcf_adjusted_age_dist[a] for a in ltcf_adjusted_age_dist])

    return n_nonltcf, ltcf_adjusted_age_dist, ltcf_adjusted_age_dist_values, ages_left_to_assign, facilities


def assign_facility_staff(datadir, location, state_location, country_location, ltcf_staff_age_min, ltcf_staff_age_max, facilities, workers_by_age_to_assign_count, potential_worker_uids_by_age, potential_worker_uids, facilities_by_uids, age_by_uid_dic, use_default=False):
    """
    Assign Long Term Care Facility staff to the generated facilities with residents.

    Args:
        datadir (string)                      : The file path to the data directory.
        location                              : name of the location
        state_location (string)               : name of the state the location is in
        country_location (string)             : name of the country the location is in
        ltcf_staff_age_min (int)              : Long term care facility staff minimum age.
        ltcf_staff_age_max (int)              : Long term care facility staff maximum age.
        facilities (list)                     : A list of lists where each sublist is a facility with the resident ages
        workers_by_age_to_assign_count (dict) : A dictionary mapping age to the count of employed individuals of that age.
        potential_worker_uids (dict)          : dictionary of potential workers mapping their id to their age
        facilities (list)                     : A list of lists where each sublist is a facility with the resident IDs
        age_by_uid_dic (dict)                 : dictionary mapping id to age for all individuals in the population
        use_default (bool)                    : If True, try to first use the other parameters to find data specific to the location under study; otherwise, return default data drawing from default_location, default_state, default_country.

    Returns:
        list: A list of lists with the facility staff IDs for each facility.
    """
    log.debug('assign_facility_staff()')
    resident_to_staff_ratio_distr = spdata.get_long_term_care_facility_resident_to_staff_ratios_distr(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)
    resident_to_staff_ratio_distr = spb.norm_dic(resident_to_staff_ratio_distr)
    resident_to_staff_ratio_brackets = spdata.get_long_term_care_facility_resident_to_staff_ratios_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    facilities_staff = []
    facilities_staff_uids = []

    sorted_ratio_keys = sorted([k for k in resident_to_staff_ratio_distr.keys()])
    ratio_array = [resident_to_staff_ratio_distr[k] for k in sorted_ratio_keys]

    staff_age_range = np.arange(ltcf_staff_age_min, ltcf_staff_age_max + 1)
    for nf, fc in enumerate(facilities):
        n_residents = len(fc)

        s = spsamp.fast_choice(ratio_array)
        s_range = resident_to_staff_ratio_brackets[s]
        resident_staff_ratio = s_range[spsamp.fast_choice(s_range)]

        n_staff = int(np.ceil(n_residents / resident_staff_ratio))
        new_staff, new_staff_uids = [], []

        for i in range(n_staff):
            a_prob = np.array([workers_by_age_to_assign_count[a] for a in staff_age_range])
            a_prob = a_prob / np.sum(a_prob)
            aindex = np.random.choice(a=staff_age_range, p=a_prob)

            uid = potential_worker_uids_by_age[aindex][0]
            potential_worker_uids_by_age[aindex].remove(uid)
            potential_worker_uids.pop(uid, None)
            workers_by_age_to_assign_count[aindex] -= 1

            new_staff.append(aindex)
            new_staff_uids.append(uid)

        facilities_staff.append(new_staff)
        facilities_staff_uids.append(new_staff_uids)

    return facilities_staff_uids


def remove_ltcf_residents_from_potential_workers(facilities_by_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid_dic):
    """
    Remove facilities residents from potential workers

    Args:
        facilities_by_uids (list)             : A list of lists, where each sublist represents a skilled nursing or long term care facility and the ids of the residents living within it
        potential_worker_uids (dict)          : dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict)   : dictionary mapping age to the list of worker ids with that age
        workers_by_age_to_assign_count (dict) : dictionary of the count of workers left to assign by age
        age_by_uid_dic (dict)                 : dictionary mapping id to age for all individuals in the population

    Returns:
        Updated dictionaries for potential worker ids, lists of potential worker
        ids mapped to age, and the number of workers left to assign by age.
    """
    log.debug('remove_ltcf_residents_from_potential_workers()')
    for nf, fc in enumerate(facilities_by_uids):
        for uid in fc:
            aindex = age_by_uid_dic[uid]
            if uid in potential_worker_uids: # pragma: no cover
                potential_worker_uids_by_age[aindex].remove(uid)
                potential_worker_uids.pop(uid, None)
                if workers_by_age_to_assign_count[aindex] > 0:
                    workers_by_age_to_assign_count[aindex] -= 1

    return potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


# Age resampling method
def ltcf_resample_age(exp_age_distr, a):
    """
    Resampling younger ages to better match data

    Args:
        exp_age_distr (dict) : age distribution
        age (int)            : age as an integer

    Returns:
        Resampled age as an integer.

    Notes:
        This is not always necessary, but is mostly used to smooth out sharp
        edges in the age distribution when spsamp.resample_age() produces too
        many of one year and under produces the surrounding ages. For example,
        new borns (0 years old) may be over produced, and 1 year olds under
        produced, so this function can be customized to correct for that. It
        is currently customized to model well the age distribution for
        Seattle, Washington.
    """
    # exp_age_distr = np.array(list(exp_age_distr_dict.values()), dtype=np.float64)
    a = spsamp.resample_age(exp_age_distr, a)
    if a == 7:
        if np.random.binomial(1, p=0.25):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 6:
        if np.random.binomial(1, p=0.25):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 5:
        if np.random.binomial(1, p=0.2):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 0:
        if np.random.binomial(1, p=0.0):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 1:
        if np.random.binomial(1, p=0.1):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 2:
        if np.random.binomial(1, p=0.0):
            a = spsamp.resample_age(exp_age_distr, a)
    if a == 4:
        if np.random.binomial(1, p=0.1):
            a = spsamp.resample_age(exp_age_distr, a)
    return a


def get_ltcf_sizes(popdict, keys_to_exclude=[]):
    """
    Get long term care facility sizes, including both residents and staff.

    Args:
        popdict (dict)         : population dictionary
        keys_to_exclude (list) : possible keys to exclude for roles in long term care facilities. See notes.

    Returns:
        dict: Dictionary of the size for each long term care facility generated.

    Notes:
        keys_to_exclude is an empty list by default, but can contain the
        different long term care facility roles: 'snf_res' for residents and
        'snf_staff' for staff. If either role is included in the parameter
        keys_to_exclude, then individuals with that value equal to 1 will not
        be counted.
    """
    log.debug('get_ltcf_sizes()')
    ltcf_sizes = dict()
    for i, person in popdict.items():
        if person['snfid'] is not None:
            ltcf_sizes.setdefault(person['snfid'], 0)

            # include facility residents
            if person['snf_res'] is not None and 'snf_res' not in keys_to_exclude:
                ltcf_sizes[person['snfid']] += 1
            # include facility staff
            elif person['snf_staff'] is not None and 'snf_staff' not in keys_to_exclude:
                ltcf_sizes[person['snfid']] += 1

    return ltcf_sizes
