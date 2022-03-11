"""
Modeling Seattle Metro Long Term Care Facilities

"""

import numpy as np
import sciris as sc
from collections import Counter
from .config import logger as log, checkmem
from . import defaults as spd
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
            expected_users_by_age[a] = np.random.binomial(ages_left_to_assign[a], ltcf_rates_by_age[a])  # use the rates to sample the number of ltcf residents by age
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


def assign_facility_staff(datadir, location, state_location, country_location, ltcf_staff_age_min, ltcf_staff_age_max, facilities, workers_by_age_to_assign_count, potential_worker_uids_by_age, potential_worker_uids, facilities_by_uids, age_by_uid, use_default=False):
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
        age_by_uid (dict)                     : dictionary mapping id to age for all individuals in the population
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


def remove_ltcf_residents_from_potential_workers(facilities_by_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid):
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
            aindex = age_by_uid[uid]
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
        different long term care facility roles: 'ltcf_res' for residents and
        'ltcf_staff' for staff. If either role is included in the parameter
        keys_to_exclude, then individuals with that value equal to 1 will not
        be counted.
    """
    log.debug('get_ltcf_sizes()')
    ltcf_sizes = dict()
    for i, person in popdict.items():
        if person['ltcfid'] is not None:
            ltcf_sizes.setdefault(person['ltcfid'], 0)

            # include facility residents
            if person['ltcf_res'] is not None and 'ltcf_res' not in keys_to_exclude:
                ltcf_sizes[person['ltcfid']] += 1
            # include facility staff
            elif person['ltcf_staff'] is not None and 'ltcf_staff' not in keys_to_exclude:
                ltcf_sizes[person['ltcfid']] += 1

    return ltcf_sizes


class LongTermCareFacility(spb.LayerGroup):
    """
    A class for individual long term care facilities and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the long term care facility
    """

    def __init__(self, ltcfid=None, resident_uids=np.array([], dtype=int), staff_uids=np.array([], dtype=int), **kwargs):
        """
        Class constructor for empty long term care facility (ltcf).

        Args:
            **ltcfid (int)             : ltcf id
            **resident_uids (np.array) : ids of ltcf members
            **staff_uids (np.array)    : ages of ltcf members
        """
        super().__init__(ltcfid=ltcfid, resident_uids=resident_uids, staff_uids=staff_uids, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a long term care facility is valid and update
        to the correct type if necessary.
        """
        for key in ['resident_uids', 'staff_uids']:
            if key in self.keys():
                try:
                    self[key] = sc.promotetoarray(self[key], dtype=int)

                except:
                    errmsg = f"Error: Could not convert ltcf key {key} to an np.array() with type int. This key only takes arrays with int values."
                    raise TypeError(errmsg)

        for key in ['ltcfid']:
            if key in self.keys():
                if not isinstance(self[key], (int, np.int32, np.int64)):
                    if self[key] is not None:
                        errmsg = f"Error: Expected type int or None for ltcf key {key}. Instead the type of this value is {type(self[key])}."
                        raise TypeError(errmsg)
        return

    @property
    def member_uids(self):
        """
        Return ids of all ltcf members: residents and staff.

        Returns:
            np.ndarray : ltcf member ids
        """
        return np.concatenate((self['resident_uids'], self['staff_uids']))

    def member_ages(self, age_by_uid):
        """
        Return ages of all ltcf members: residents and staff.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : ltcf member ages
        """
        return np.concatenate((self.resident_ages(age_by_uid), self.staff_ages(age_by_uid)))

    def __len__(self):
        """Return the length as the number of members in the ltcf."""
        return len(self.member_uids)

    def resident_ages(self, age_by_uid):
        """
        Return ages of ltcf residents.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : ltcf resident ages
        """
        return super().member_ages(age_by_uid, self['resident_uids'])

    def staff_ages(self, age_by_uid):
        """
        Return ages of ltcf staff.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : ltcf staff ages
        """
        return super().member_ages(age_by_uid, self['staff_uids'])


def get_ltcf(pop, ltcfid):
    """
    Return ltcf with id: ltcfid.

    Args:
        pop (sp.Pop) : population
        ltcfid (int) : ltcf id number

    Returns:
        sp.LongTermCareFacility: A populated ltcf.
    """
    if not isinstance(ltcfid, int):
        raise TypeError(f"ltcfid must be an int. Instead supplied wpid with type: {type(ltcfid)}.")
    if len(pop.ltcfs) <= ltcfid:
        raise IndexError(f"Ltcf id (ltcfid): {ltcfid} out of range. There are {len(pop.ltcfs)} ltcfs stored in this object.")
    return pop.ltcfs[ltcfid]


def add_ltcf(pop, ltcf):
    """
    Add a ltcf to the list of ltcfs.

    Args:
        pop (sp.Pop)                   : population
        ltcf (sp.LongTermCareFacility) : ltcf with at minimum the ltcfid, resident_uids and staff_uids.
    """
    if not isinstance(ltcf, LongTermCareFacility):
        raise ValueError('ltcf is not a sp.LongTermCareFacility object.')

    # ensure ltcfid to match the index in the list
    if ltcf['ltcfid'] != len(pop.ltcfs):
        ltcf['ltcfid'] = len(pop.ltcfs)
    pop.ltcfs.append(ltcf)
    pop.n_ltcfs = len(pop.ltcfs)
    return


def initialize_empty_ltcfs(pop, n_ltcfs=None):
    """
    Array of empty ltcfs.

    Args:
        pop (sp.Pop)  : population
        n_ltcfs (int) : the number of ltcfs to initialize
    """
    if n_ltcfs is not None and isinstance(n_ltcfs, int):
        pop.n_ltcfs = n_ltcfs
    else:
        pop.n_ltcfs = 0

    pop.ltcfs = [LongTermCareFacility() for nl in range(pop.n_ltcfs)]
    return


def populate_ltcfs(pop, resident_lists, staff_lists):
    """
    Populate all of the ltcfs. Store each ltcf at the index corresponding to it's ltcfid.

    Args:
        pop (sp.Pop)          : population
        residents_list (list) : list of lists where each sublist represents a ltcf and contains the ids of the residents
        staff_lists (list)    : list of lists where each sublist represents a ltcf and contains the ids of the staff
    """
    initialize_empty_ltcfs(pop, len(resident_lists))

    log.debug("Populating ltcfs.")

    # now populate ltcfs
    for nl, residents in enumerate(resident_lists):
        lf = []
        lf.extend(residents)
        lf.extend(staff_lists[nl])
        kwargs = dict(ltcfid=nl,
                      resident_uids=residents,
                      staff_uids=staff_lists[nl],
                      )
        ltcf = LongTermCareFacility()
        ltcf.set_layer_group(**kwargs)
        pop.ltcfs[ltcf['ltcfid']] = sc.dcp(ltcf)

    return
