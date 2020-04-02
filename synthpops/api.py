import os
import numpy as np
import sciris as sc
import synthpops as sp

__all__ = ['popsize_choices', 'make_population', 'rehydrate']

# Put this here so it's accessible as sp.api.choices
popsize_choices = [5000,
                   10000,
                   20000,
                   50000,
                   100000,
                   122000,
                ]


def make_population(n=None, max_contacts=None, as_objdict=False):
    '''
    Make a full population network including both people (ages, sexes) and contacts.

    Args:
        n (int): number of people to create
        max_contacts (dict): dictionary for maximum number of contacts per layer: keys must be 'S' (school) and/or 'W' (work)
        as_objdict (bool): whether to return as an object dictionary -- easier to work with, but slower

    Returns:
        network (dict): a dictionary of the full population with ages and connections

    '''

    default_n = 20000

    if n is None: n = default_n
    n = int(n)
    if n not in popsize_choices:
        choicestr = ', '.join([str(choice) for choice in popsize_choices])
        errormsg = f'Number of people must be one of {choicestr}, not {n}'
        raise ValueError(errormsg)


    filename = f'synthpop_{n}.pop'
    filepath = sc.makefilepath(folder=sp.localdatadir, filename=filename)
    if not os.path.isfile(filepath):
        errormsg = f'Path {filepath} not found!'
        raise FileNotFoundError(errormsg)

    data = sc.loadobj(filepath)

    sc.tic()
    population = rehydrate(data, max_contacts=max_contacts)

    # Change types
    if as_objdict:
        population = sc.objdict(population)
    for key,person in population.items():
        for layerkey in population[key]['contacts'].keys():
            population[key]['contacts'][layerkey] = list(population[key]['contacts'][layerkey])
        if as_objdict:
            population[key] = sc.objdict(population[key])
            population[key]['contacts'] = sc.objdict(population[key]['contacts'])

    return population


def rehydrate(data, max_contacts=None):
    default_max_contacts = {'H':0, 'S':20, 'W':10}
    max_contacts = sc.mergedicts(default_max_contacts, max_contacts)

    popdict = sc.dcp(data['popdict'])
    mapping = {'H':'households', 'S':'schools', 'W':'workplaces'}
    for key,label in mapping.items():
        for r in data[label]: # House, school etc
            for uid in r:
                current_contacts = len(popdict[uid]['contacts'][key])
                if max_contacts[key]:  # 0 for unlimited
                    to_select = max_contacts[key] - current_contacts
                    if to_select <= 0:  # already filled list from other actors
                        continue
                    contacts = np.random.choice(r, size=to_select)
                else:
                    contacts = r
                for c in contacts:
                    if c == uid:
                        continue
                    if c in popdict[uid]['contacts'][key]:
                        continue
                    popdict[uid]['contacts'][key].add(c)
                    popdict[c]['contacts'][key].add(uid)
    return popdict