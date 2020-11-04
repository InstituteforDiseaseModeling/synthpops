"""
This module provides the layer for communicating with the agent-based model Covasim.
"""

import synthpops as sp
from .config import logger as log


def make_population(*args, **kwargs):
    '''
    Interface to sp.Pop().to_dict(). Included for backwards compatibility.
    '''
    log.debug('make_population()')

    # Heavy lift 1: make the contacts and their connections
    log.debug('Generating a new population...')
    pop = sp.Pop(*args, **kwargs)

    population = pop.to_dict()

    log.debug('make_population(): done.')
    return population
