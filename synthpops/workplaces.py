from copy import deepcopy
from collections import Counter
import sciris as sc
import numpy as np
from . import base as spb
from . import sampling as spsamp
from .config import logger as log
from . import defaults


__all__ = ['count_employment_by_age', 'get_workplace_sizes',
           'get_employment_rates_by_age',
           'get_generated_workplace_size_distribution',
           'Workplace',
           ]


class Workplace(spb.LayerGroup):
    """
    A class for individual workplaces and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the workplace
    """

    def __init__(self, wpid=None, **kwargs):
        """
        Class constructor for empty workplace.

        Args:
            **wpid (int)             : workplace id
            **member_uids (np.array) : ids of workplace members
        """
        # set up default workplace values
        super().__init__(wpid=wpid, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a workplace is valid and update
        to the correct type if necessary.
        """
        super().validate(layer_str='workplace')
        return


__all__ += ['get_workplace', 'add_workplace', 'initialize_empty_workplaces', 'populate_workplaces']


def get_workplace(pop, wpid):
    """
    Return workplace with id: wpid.

    Args:
        pop (sp.Pop) : population
        wpid (int)   : workplace id number

    Returns:
        sp.Workplace: A populated workplace.
    """
    if not isinstance(wpid, int):
        raise TypeError(f"wpid must be an int. Instead supplied wpid with type: {type(wpid)}.")
    if len(pop.workplaces) <= wpid:
        raise IndexError(f"Workplace id (wpid): {wpid} out of range. There are {len(pop.workplaces)} workplaces stored in this object.")
    return pop.workplaces[wpid]


def add_workplace(pop, workplace):
    """
    Add a workplace to the list of workplaces.

    Args:
        pop (sp.Pop)             : population
        workplace (sp.Workplace) : workplace with at minimum the wpid and member_uids.
    """
    if not isinstance(workplace, Workplace):
        raise ValueError('workplace is not a sp.Workplace object.')

    # ensure wpid to match the index in the list
    if workplace['wpid'] != len(pop.workplaces):
        workplace['wpid'] = len(pop.workplaces)
    pop.workplaces.append(workplace)
    pop.n_workplaces = len(pop.workplaces)
    return


def initialize_empty_workplaces(pop, n_workplaces=None):
    """
    Array of empty workplaces.

    Args:
        pop (sp.Pop)       : population
        n_workplaces (int) : the number of workplaces to initialize
    """
    if n_workplaces is not None and isinstance(n_workplaces, int):
        pop.n_workplaces = n_workplaces
    else:
        pop.n_workplaces = 0

    pop.workplaces = [Workplace() for nw in range(pop.n_workplaces)]
    return


def populate_workplaces(pop, workplaces):
    """
    Populate all of the workplaces. Store each workplace at the index corresponding to it's wpid.

    Args:
        pop (sp.Pop)      : population
        workplaces (list) : list of lists where each sublist represents a workplace and contains the ids of the workplace members

    Notes:
        If number of workplaces (n) is fewer than existing workplaces, it will only replace the first n workplaces. Otherwise the
        existing workplaces will be overwritten by the input workplaces.
    """
    # make sure there are enough workplaces
    initialize_empty_workplaces(pop, len(workplaces))

    log.debug("Populating workplaces.")

    # now populate workplaces
    for nw, wp in enumerate(workplaces):
        kwargs = dict(wpid=nw,
                      member_uids=wp,
                      )
        workplace = Workplace()
        workplace.set_layer_group(**kwargs)
        pop.workplaces[workplace['wpid']] = sc.dcp(workplace)

    return


def get_uids_potential_workers(student_uid_lists, employment_rates, age_by_uid):
    """
    Get IDs for everyone who could be a worker by removing those who are students and those who can't be employed officially.

    Args:
        student_uid_lists (list) : A list of lists where each sublist represents a school with the IDs of students in the school.
        employment_rates (dict)  : The employment rates by age.
        age_by_uid (dict)        : A dictionary mapping ID to age for individuals in the population.

    Returns:
        A dictionary of potential workers mapping their ID to their age, a dictionary mapping age to the list of IDs for potential
        workers with that age, and a dictionary mapping age to the count of potential workers left to assign to a workplace for that age.
    """
    log.debug('get_uids_potential_workers()')
    potential_worker_uids = deepcopy(age_by_uid)
    potential_worker_uids_by_age = {}
    potential_worker_ages_left_count = {}

    for a in range(101):
        if a >= 15:
            potential_worker_uids_by_age[a] = []
            potential_worker_ages_left_count[a] = 0

    # remove students from any potential workers since the model assumes student and worker status are exclusive
    for students in student_uid_lists:
        for uid in students:
            potential_worker_uids.pop(uid, None)

    for uid in age_by_uid:
        if age_by_uid[uid] not in employment_rates:
            potential_worker_uids.pop(uid, None)

    for uid in potential_worker_uids:
        ai = potential_worker_uids[uid]

        # potential_worker_uid[uid] may generate persons who are not valid working age
        # This will cause a 'key' error in potential__worker_uids_by_age
        # Since potential_worker_uids_age keys are valid work ages, skip invalid workers
        if ai in potential_worker_uids_by_age.keys():
            potential_worker_uids_by_age[ai].append(uid)
            potential_worker_ages_left_count[ai] += 1

    # shuffle workers around!
    for ai in potential_worker_uids_by_age:
        np.random.shuffle(potential_worker_uids_by_age[ai])

    return potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count


def generate_workplace_sizes(workplace_size_distr_by_bracket, workplace_size_brackets, workers_by_age_to_assign_count):
    """
    Given a number of individuals employed, generate a list of workplace sizes to place everyone in a workplace.

    Args:
        workplace_size_distr_by_bracket (dict) : The distribution of binned workplace sizes.
        worlplace_size_brackets (dict)         : A dictionary of workplace size brackets.
        workers_by_age_to_assign_count (dict)  : A dictionary mapping age to the count of employed individuals of that age.

    Returns:
        A list of workplace sizes.
    """
    nworkers = np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count])

    # normalize workplace_size_distr_by_bracket because it's likely a count rather than distribution
    workplace_size_distr_by_bracket = spb.norm_dic(workplace_size_distr_by_bracket)

    sorted_brackets = sorted(workplace_size_brackets.keys())
    prob_by_sorted_brackets = [workplace_size_distr_by_bracket[b] for b in sorted_brackets]

    workplace_sizes = []

    while nworkers > 0:
        size_bracket = np.random.choice(sorted_brackets, p=prob_by_sorted_brackets)
        size = np.random.choice(workplace_size_brackets[size_bracket])
        nworkers -= size
        workplace_sizes.append(size)
    if nworkers < 0:
        workplace_sizes[-1] = workplace_sizes[-1] + nworkers
    np.random.shuffle(workplace_sizes)
    return workplace_sizes


def get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count, uids_by_age):
    """
    Get the number of people to assign to a workplace by age using those left who can potentially go to work and employment rates by age.

    Args:
        employment_rates (dict)                 : A dictionary of employment rates by age.
        potential_worker_ages_left_count (dict) : A dictionary of the count of workers to assign by age.
        uids_by_age (dict)                      : A dictionary mapping age to the list of ids with that age.

    Returns:
        A dictionary with a count of workers to assign to a workplace.
    """
    log.debug('get_workers_by_age_to_assign()')
    workers_by_age_to_assign_count = dict.fromkeys(np.arange(101), 0)
    for a in potential_worker_ages_left_count:
        if a in employment_rates:
            try:
                c = int(employment_rates[a] * len(uids_by_age[a]))
            except:
                c = 0
            number_of_people_who_can_be_assigned = min(c, potential_worker_ages_left_count[a])
            workers_by_age_to_assign_count[a] = number_of_people_who_can_be_assigned

    return workers_by_age_to_assign_count


def assign_rest_of_workers(workplace_sizes, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count, age_by_uid, age_brackets, age_by_brackets, contact_matrices):
    """
    Assign the rest of the workers to non-school workplaces.

    Args:
        workplace_sizes (list)                : list of workplace sizes
        potential_worker_uids (dict)          : dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict)   : dictionary mapping age to the list of worker ids with that age
        workers_by_age_to_assign_count (dict) : dictionary of the count of workers left to assign by age
        age_by_uid (dict)                     : dictionary mapping id to age for all individuals in the population
        age_brackets (dict)                   : dictionary mapping age bracket keys to age bracket range
        age_by_brackets (dict)                : dictionary mapping age to the age bracket range it falls in
        contact_matrices (dict)               : dictionary of age specific contact matrix for different physical contact settings

    Returns:
        List of lists where each sublist is a workplace with the ages of workers, list of lists where each sublist is a workplace with the ids of workers,
        dictionary of potential workers left mapping id to age, dictionary mapping age to a list of potential workers left of that age, dictionary
        mapping age to the count of workers left to assign.
    """
    log.debug('assign_rest_of_workers()')
    workplace_age_lists = []
    workplace_uid_lists = []
    worker_age_keys = workers_by_age_to_assign_count.keys()
    sorted_worker_age_keys = sorted(worker_age_keys)

    # make a copy of the workplace matrix to sample from and modify as people get placed into workplaces and removed from the pool of potential workers
    w_contact_matrix = contact_matrices['W'].copy()

    # off turn likelihood to meet those unemployed in the workplace because the matrices are not an exact match for the population under study
    for b in age_brackets:
        workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[b]]
        number_of_workers_left_in_bracket = np.sum(workers_left_in_bracket)
        if number_of_workers_left_in_bracket == 0:
            b = min(b, w_contact_matrix.shape[1] - 1)  # Ensure it doesn't go past the end of the array
            w_contact_matrix[:, b] = 0

    for n, size in enumerate(workplace_sizes):
        workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        if sum(workers_by_age_to_assign_distr.values()) == 0:
            break
        if sum([len(v) for v in potential_worker_uids_by_age.values()]) == 0:
            break
        new_work, new_work_uids = [], []

        a_prob = [workers_by_age_to_assign_count[a] for a in sorted_worker_age_keys]
        a_prob = np.array(a_prob)
        a_prob = a_prob / np.sum(a_prob)

        achoice = np.random.choice(a=sorted_worker_age_keys, p=a_prob)
        aindex = achoice

        uid = potential_worker_uids_by_age[aindex][0]
        potential_worker_uids_by_age[aindex].remove(uid)
        potential_worker_uids.pop(uid, None)
        workers_by_age_to_assign_count[aindex] -= 1
        workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        new_work.append(aindex)
        new_work_uids.append(uid)

        bindex = age_by_brackets[aindex]
        bindex = min(bindex, w_contact_matrix.shape[0] - 1)  # Ensure it doesn't go past the end of the array
        b_prob = w_contact_matrix[bindex, :]
        sum_b_prob = np.sum(b_prob)
        if sum_b_prob > 0: # pragma: no cover
            b_prob = b_prob / sum_b_prob

        if size > len(potential_worker_uids) - 1: # pragma: no cover
            size = len(potential_worker_uids) - 1
        workers_left_count = np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count])
        if size > workers_left_count:
            size = workers_left_count + 1

        # not enough people left over to try to match age mixing patterns in the last workplace so grab everyone who will get placed in order
        if len(potential_worker_uids) <= size or workers_left_count <= size:
            for ai in workers_by_age_to_assign_count:
                for i in range(workers_by_age_to_assign_count[ai]):  # do not change this during the loop but afterwards, and if 0 then no one will be placed
                    uid = potential_worker_uids_by_age[ai][0]
                    new_work.append(ai)
                    new_work_uids.append(uid)
                    potential_worker_uids_by_age[ai].remove(uid)
                    potential_worker_uids.pop(uid, None)
                workers_by_age_to_assign_count[ai] = 0  # set to zero now that everyone will be placed in this last workplace
            workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)
        else:
            for i in range(1, size):

                bi = spsamp.fast_choice(b_prob)

                workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi] if len(potential_worker_uids_by_age[a]) > 0]

                if np.sum(b_prob): # pragma: no cover
                    loop_b_prob = sc.dcp(b_prob)  # Make a copy to avoid overwriting the original
                    while np.sum(workers_left_in_bracket) == 0:
                        loop_b_prob[bi] = 0  # Don't pick the same bracket ever again
                        bi = spsamp.fast_choice(loop_b_prob)
                        workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi] if len(potential_worker_uids_by_age[a]) > 0]
                    a_prob = [workers_by_age_to_assign_count[a] for a in age_brackets[bi]]
                    ai = age_brackets[bi][spsamp.fast_choice(a_prob)]

                    uid = potential_worker_uids_by_age[ai][0]
                    new_work.append(ai)
                    new_work_uids.append(uid)
                    potential_worker_uids_by_age[ai].remove(uid)
                    potential_worker_uids.pop(uid, None)
                    workers_by_age_to_assign_count[ai] -= 1
                    workers_by_age_to_assign_distr = spb.norm_dic(workers_by_age_to_assign_count)

                # if there's no one left in the bracket, then you should turn this bracket off in the contact matrix
                workers_left_in_bracket = [workers_by_age_to_assign_count[a] for a in age_brackets[bi]]
                if np.sum(workers_left_in_bracket) == 0:
                    w_contact_matrix[:, bi] = 0.
                    # since the matrix was modified, calculate the bracket probabilities again
                    b_prob = w_contact_matrix[bindex, :]
                    if np.sum(b_prob) > 0: # pragma: no cover
                        b_prob = b_prob / np.sum(b_prob)

        log.debug(f'  Progress: {n}, {Counter(new_work)}')
        workplace_age_lists.append(new_work)
        workplace_uid_lists.append(new_work_uids)
    return workplace_age_lists, workplace_uid_lists, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


def count_employment_by_age(popdict):
    """
    Get employment count by age for workers in the popdict. Workers can be in
    different possible layers: as staff in long term care facilities (LTCF),
    as teachers or staff in schools (S), or as workers in other workplaces (W).

    Args:
        popdict (dict) : population dictionary

    Returns:
        dict: Dictionary of the count of employed people by age in popdict.
    """
    employment_count_by_age = dict.fromkeys(np.arange(0, defaults.settings.max_age), 0)
    for i, person in popdict.items():
        if person['ltcf_staff'] or person['sc_teacher'] or person['sc_staff'] or person['wpid']:
            employment_count_by_age[person['age']] += 1

    return employment_count_by_age


def get_employment_rates_by_age(employment_count_by_age, age_count):
    """
    Get employment rates by age.

    Args:
        employment_count_by_age (dict) : dictionary of the count of employed people
        age_count (dict)               : dictionary of the age count

    Returns:
        dict: Dictionary of the employment rates by age.
    """
    return {a: employment_count_by_age[a] / age_count[a] if age_count[a] > 0 else 0 for a in sorted(age_count.keys())}


def get_workplace_sizes(popdict):
    """
    Get workplace sizes of regular workplaces in popdict. This only includes
    workplaces that are not long term care facilities (LTCF) or schools (S).

    Args:
        popdict (dict) : population dictionary

    Returns:
        dict: Dictionary of the generated workplace sizes for each regular workplace.
    """
    workplace_sizes = dict()
    for i, person in popdict.items():
        if person['wpid'] is not None:
            workplace_sizes.setdefault(person['wpid'], 0)
            workplace_sizes[person['wpid']] += 1
            # workplace_sizes.setdefault(person['wpid'], dict())  # use when workplace types by industry are included
            # workplace_sizes[person['wpid']].setdefault('employed', 0)
            # workplace_sizes[person['wpid']]['employed'] += 1

    return workplace_sizes


def get_generated_workplace_size_distribution(workplace_sizes, bins):
    """
    Get workplace size distribution.

    Args:
        workplace_sizes (dict): generated workplace sizes by workplace id (wpid)
        bins (list) : workplace size bins

    Returns:
        dict: Dictionary of generated workplace size distribution.
    """
    generated_workplace_sizes = list(workplace_sizes.values())
    hist, bins = np.histogram(generated_workplace_sizes, bins=bins, density=0)
    if sum(generated_workplace_sizes) > 0:
        generated_workplace_size_dist = {i: hist[i] / sum(hist) for i in range(len(hist))}
    else:
        generated_workplace_size_dist = {i: hist[i] for i in range(len(hist))}

    return generated_workplace_size_dist
