"""
Modeling Seattle Metro Long Term Care Facilities

"""

import numpy as np
import sciris as sc
from collections import Counter
from .config import logger as log, checkmem
from . import defaults as spd
from . import sampling as spsamp
from . import households as sphh
from . import data_distributions as spdata
from . import base as spb


def generate_ltcfs(n, with_facilities, datadir, country_location, state_location, location, use_default, smooth_ages, window_length):
    """
    Generate residents living in long term care facilities and their ages.

    Args:
        n (int)                   : The number of people to create.
        with_facilities (bool)    : If True, create long term care facilities, currently only available for locations in the US.
        datadir (string)          : The file path to the data directory.
        country_location (string) : name of the country the location is in
        state_location (string)   : name of the state the location is in
        location                  : name of the location
        use_default (bool)        : If True, try to first use the other parameters to find data specific to the location under study; otherwise, return default data drawing from default_location, default_state, default_country.
        smooth_ages (bool)        : If True, use smoothed out age distribution.
        window_length (int)       : length of window over which to average or smooth out age distribution

    Returns:
        The number of people expected to live outside long term care facilities,
        age_brackets, age_by_brackets dictionary, age distribution adjusted for
        long term care facility residents already sampled, and facilities with
        people living in them.

    """
    # Initialize outputs and load location age distribution
    facilities = []
    age_distr = spdata.read_age_bracket_distr(datadir, country_location=country_location, state_location=state_location, location=location)
    age_brackets = spdata.get_census_age_brackets(datadir, country_location=country_location, state_location=state_location, location=location)
    age_by_brackets = spb.get_age_by_brackets(age_brackets)

    expected_age_distr = dict.fromkeys(age_by_brackets.keys(), 0)
    for a in expected_age_distr:
        b = age_by_brackets[a]
        expected_age_distr[a] = age_distr[b] / len(age_brackets[b])

    if smooth_ages:
        smoothed_age_distr = spdata.get_smoothed_single_year_age_distr(datadir, location=location,
                                                                       state_location=state_location,
                                                                       country_location=country_location,
                                                                       window_length=window_length)
        expected_age_distr = smoothed_age_distr.copy()

    n = int(n)
    expected_users_by_age = dict.fromkeys(age_by_brackets.keys(), 0)

    max_age = max(age_by_brackets.keys())

    # If not using facilities, skip everything here
    if with_facilities:
        # Get long term care facilities data at the state level
        ltcf_rates_by_age = spdata.get_long_term_care_facility_use_rates(datadir, state_location=state_location, country_location=country_location)

        # for the population of size n, calculate the number of people at each age expected to live in long term care facilities
        for a in expected_users_by_age:
            b = age_by_brackets[a]
            expected_users_by_age[a] = int(np.ceil(n * expected_age_distr[a] * ltcf_rates_by_age[a]))

        # make a list of all resident ages
        all_residents = []
        for a in expected_users_by_age:
            all_residents.extend([a] * expected_users_by_age[a])

        np.random.shuffle(all_residents)  # randomly shuffle ages

        # how big are long term care facilities
        resident_size_distr = spdata.get_long_term_care_facility_residents_distr(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)
        resident_size_distr = spb.norm_dic(resident_size_distr)
        resident_size_brackets = spdata.get_long_term_care_facility_residents_distr_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

        size_bracket_keys = sorted(resident_size_distr.keys())
        size_distr_array = [resident_size_distr[k] for k in size_bracket_keys]
        while len(all_residents) > 0:

            s = spsamp.fast_choice(size_distr_array)
            s_range = resident_size_brackets[s]
            size = np.random.choice(s_range)

            if size > len(all_residents):
                size = len(all_residents)

            new_facility = all_residents[:size]
            facilities.append(new_facility)
            all_residents = all_residents[size:]

        # adjust age distribution

        ltcf_adjusted_age_distr_dict = dict.fromkeys(age_by_brackets.keys(), 0)
        for a in range(max_age + 1):
            ltcf_adjusted_age_distr_dict[a] = expected_age_distr[a]
            ltcf_adjusted_age_distr_dict[a] -= float(expected_users_by_age[a]) / n  # remove long term care facility residents from the age distribution

        ltcf_adjusted_age_distr_array = np.array([ltcf_adjusted_age_distr_dict[a] for a in range(max_age + 1)])  # make an array of the age distribution

        n_nonltcf = int(n - sum([len(f) for f in facilities]))  # remove those placed as residents in long term care facilities

    else:
        n_nonltcf = n
        ltcf_adjusted_age_distr_dict = dict.fromkeys(age_by_brackets.keys(), 0)
        for a in range(max_age + 1):
            ltcf_adjusted_age_distr_dict[a] = expected_age_distr[a]
        ltcf_adjusted_age_distr_array = np.array([ltcf_adjusted_age_distr_dict[a] for a in range(max_age + 1)])  # make an array of the age distribution

    return n_nonltcf, age_brackets, age_by_brackets, ltcf_adjusted_age_distr_array, facilities


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
    """
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


# Household construction methods

def generate_larger_households_method_1(size, hh_sizes, hha_by_size_counts, hha_brackets, cm_age_brackets, cm_age_by_brackets, contact_matrices, single_year_age_distr):
    """
    Generate ages of those living in households of greater than one individual.
    Reference individual is sampled conditional on the household size. All other
    household members have their ages sampled conditional on the reference
    person's age and the age mixing contact matrix in households for the
    population under study.

    Args:
        size (int)                   : The household size.
        hh_sizes (array)             : The count of household size s at index s-1.
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        cm_age_brackets (dict)       : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets (dict)    : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrices (dict)      : A dictionary of the age-specific contact matrix for different physical contact settings.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        An array of households for size ``size`` where each household is a row
        and the values in the row are the ages of the household members. The
        first age in the row is the age of the reference individual.
    """
    log.debug('generate_larger_households()')
    ya_coin = 0.15  # This is a placeholder value. Users will need to change to fit whatever population you are working with

    homes = np.zeros((hh_sizes[size-1], size), dtype=int)

    for h in range(hh_sizes[size-1]):

        hha = sphh.generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, size, single_year_age_distr)

        homes[h][0] = hha

        b = cm_age_by_brackets[hha]
        b = min(b, contact_matrices['H'].shape[0]-1)  # Ensure it doesn't go past the end of the array
        b_prob = contact_matrices['H'][b, :]

        for n in range(1, size):
            bi = spsamp.sample_single_arr(b_prob)
            ai = spsamp.sample_from_range(single_year_age_distr, cm_age_brackets[bi][0], cm_age_brackets[bi][-1])

            """ The following is an example of how you may resample from an age range that is over produced and instead
                sample ages from an age range that is under produced in your population. This kind of customization may
                be necessary when your age mixing matrix and the population you are interested in modeling differ in
                important but subtle ways. For example, generally household age mixing matrices reflect mixing patterns
                for households composed of families. This means household age mixing matrices do not generally cover
                college or university aged individuals living together. Without this customization, this algorithm tends
                to under produce young adults. This method also has a tendency to underproduce the elderly, and does not
                explicitly model the elderly living in nursing homes. Customizations like this should be considered in
                context of the specific population and culture you are trying to model. In some cultures, it is common to
                live in non-family households, while in others family households are the most common and include
                multi-generational family households. If you are unsure of how to proceed with customizations please
                take a look at the references listed in the overview documentation for more information.
            """
            if ai > 5 and ai <= 20:  # This a placeholder range. Users will need to change to fit whatever population you are working with
                if np.random.binomial(1, ya_coin):
                    ai = spsamp.sample_from_range(single_year_age_distr, 25, 32)  # This is a placeholder range. Users will need to change to fit whatever populaton you are working with

            # ai = spsamp.resample_age(single_year_age_distr, ai)
            ai = ltcf_resample_age(single_year_age_distr, ai)

            homes[h][n] = ai

    return homes


def generate_all_households_method_1(N, hh_sizes, hha_by_size_counts, hha_brackets, cm_age_brackets, cm_age_by_brackets, contact_matrices, single_year_age_distr):
    """
    Generate the ages of those living in households together. First create households of people living alone, then larger households.
    For households larger than 1, a reference individual's age is sampled conditional on the household size, while all other household
    members have their ages sampled conditional on the reference person's age and the age mixing contact matrix in households for the
    population under study.

    Args:
        N (int)                      : The number of people in the population.
        hh_sizes (array)             : The count of household size s at index s-1.
        hha_by_size_counts (matrix)  : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)          : The age brackets for the heads of household.
        cm_age_brackets (dict)       : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets (dict)    : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrices (dict)      : The dictionary of the age-specific contact matrix for different physical contact settings.
        single_year_age_distr (dict) : The age distribution.

    Returns:
        An array of all households where each household is a row and the values in the row are the ages of the household members.
        The first age in the row is the age of the reference individual. Households are randomly shuffled by size.

    Note:
        This method is not guaranteed to model the population age distribution well automatically. The method called
        inside, generate_larger_households_method_1 uses the method ltcf_resample_age to fit Seattle, Washington populations with long term
        care facilities generated. For a method that matches the age distribution well for populations in general, please use generate_all_households_methods_2.

    """

    homes_dic = dict()
    homes_dic[1] = sphh.generate_living_alone(hh_sizes, hha_by_size_counts, hha_brackets, single_year_age_distr)
    # remove living alone from the distribution to choose from!
    for h in homes_dic[1]:
        single_year_age_distr[h[0]] -= 1.0/N

    # generate larger households and the ages of people living in them
    for s in range(2, len(hh_sizes) + 1):
        homes_dic[s] = generate_larger_households_method_1(s, hh_sizes, hha_by_size_counts, hha_brackets, cm_age_brackets, cm_age_by_brackets, contact_matrices, single_year_age_distr)

    homes = []
    for s in homes_dic:
        homes += list(homes_dic[s])

    np.random.shuffle(homes)
    return homes_dic, homes


def generate_all_households_method_2(n_nonltcf, hh_sizes, hha_by_size, hha_brackets, cm_age_brackets, cm_age_by_brackets, contact_matrices, ltcf_adjusted_age_distr):
    """
    Generate the ages of those living in households together. First create households of people living alone, then larger households.
    For households larger than 1, a reference individual's age is sampled conditional on the household size, while all other household
    members have their ages sampled conditional on the reference person's age and the age mixing contact matrix in households
    for the population under study. Fix the count of ages in the population before placing individuals in households so that the
    age distribution of the generated population is fixed to closely match the age distribution from data on the population.

    Args:
        n_nonltcf (int)                : The number of people in the population not living in long term care facilities.
        hh_sizes (array)               : The count of household size s at index s-1.
        hha_by_size_counts (matrix)    : A matrix in which each row contains the age distribution of the reference person for household size s at index s-1.
        hha_brackets (dict)            : The age brackets for the heads of household.
        cm_age_brackets (dict)         : The dictionary mapping age bracket keys to age bracket range matching the household contact matrix.
        cm_age_by_brackets (dict)      : The dictionary mapping age to the age bracket range it falls within matching the household contact matrix.
        contact_matrices (dict)        : The dictionary of the age-specific contact matrix for different physical contact settings.
        ltcf_adjusted_age_distr (dict) : The age distribution.

    Returns:
        An array of all households where each household is a row and the values in the row are the ages of the household members.
        The first age in the row is the age of the reference individual. Households are randomly shuffled by size.
    """
    nonlctf_age_distr = ltcf_adjusted_age_distr / ltcf_adjusted_age_distr.sum()  # use this to generate the rest of the ages
    nonltcf_age_count = sphh.generate_age_count(n_nonltcf, nonlctf_age_distr)
    homes_dic = dict()
    homes_dic[1] = sphh.generate_living_alone_method_2(hh_sizes, hha_by_size, hha_brackets, nonltcf_age_count)

    living_alone_ages = [homes_dic[1][h][0] for h in range(len(homes_dic[1]))]
    living_alone_age_count = Counter(living_alone_ages)

    ages_left_to_assign = dict.fromkeys(np.arange(len(ltcf_adjusted_age_distr)))

    # remove those already placed in households on their own
    for a in ages_left_to_assign:
        ages_left_to_assign[a] = nonltcf_age_count[a] - living_alone_age_count[a]

    # create array of expected household sizes  larger than out of order so that running out of individuals to place by age is not systemically as issue for larger household sizes

    max_hh_size = len(hh_sizes)

    larger_hh_size_array = sphh.generate_larger_household_sizes(hh_sizes)

    for hs in range(2, max_hh_size + 1):
        homes_dic[hs] = []

    # go through every household and assign age of the head of the household
    larger_hha_chosen, ages_left_to_assign = sphh.generate_larger_households_head_ages(larger_hh_size_array, hha_by_size, hha_brackets, ages_left_to_assign)
    larger_hha_count = Counter(larger_hha_chosen)

    # make copy of the household matrix that you can modify to help with sampling
    household_matrix = contact_matrices['H'].copy()

    homes_dic, ages_left_to_assign = sphh.generate_larger_households_method_2(larger_hh_size_array, larger_hha_chosen, hha_brackets, cm_age_brackets, cm_age_by_brackets, household_matrix, ages_left_to_assign, homes_dic)

    homes = sphh.get_all_households(homes_dic)
    return homes_dic, homes


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
                    errmsg = f"Could not convert ltcf key {key} to an np.array() with type int. This key only takes arrays with int values."
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

        Returns:
            np.ndarray : ltcf member ages
        """
        return np.concatenate((self.member_ages(age_by_uid, self['resident_uids']), self.member_ages(age_by_uid, self['staff_uids'])))

    def __len__(self):
        """Return the length as the number of members in the ltcf."""
        return len(self.member_uids)

    def resident_ages(self, age_by_uid):
        return self.member_ages(age_by_uid, self['resident_uids'])

    def staff_ages(self, age_by_uid):
        return self.member_ages(age_by_uid, self['staff_uids'])


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
