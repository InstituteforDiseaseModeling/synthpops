"""
A few examples of how to make synthetic populations with random contact networks.
Populations can have demographics (age, sex) from data or randomly generated demographics.
"""

import synthpops as sp
import sciris as sc
import numpy as np
import os


def make_popdict_of_random_people(n=1e4):
    """
    Make a popdict of n people, age and sex assigned randomly, not informed by data. Age range from 0 to 100.
    """
    popdict = sp.make_popdict(n=n)

    return popdict


def make_popdict_of_people(n=1e4,location='seattle_metro',state_location='Washington',country_location='usa'):
    """
    Make a popdict of n people, age and sex sampled from Seattle Metro demographics.
    """
    popdict = sp.make_popdict(n=n,location=location,state_location=state_location,country_location=country_location,use_demography=True)

    return popdict


def make_popdict_with_supplied_ages(datadir,n=1e4,location='seattle_metro',state_location='Washington',country_location='usa',use_demography=True):
    """
    Make a popdict of n people, with ages supplied. 
    """
    if use_demography: # get ages from demographic data and supply them to popdict
        ages = sp.get_age_n(datadir,n=n,location=location,state_location=state_location,country_location=country_location)
        popdict = sp.make_popdict(n=n,ages=ages,location=location,state_location=state_location,country_location=country_location)
    
    else: # supply any ages you want between and this will populate them in the popdict object
        min_age,max_age = 0,100
        ages = np.random.randint(min_age,max_age+1,size = n) # supply any distribution you like
        popdict = sp.make_popdict(n=n,ages=ages)

    return popdict


def make_random_contacts(n=1e4,location='seattle_metro',state_location='Washington',country_location='usa',average_degree=30,verbose=False):
    """
    Make a popdict of n people, age and sex sampled from Seattle Metro demographics but random contacts.
    Network created is an Erdos-Renyi network with average degree of 30.
    """
    popdict = sp.make_popdict(n=n,location=location,state_location=state_location,country_location=country_location,use_demography=True)

    network_distr_args = {'average_degree': average_degree}
    contacts = sp.make_contacts(popdict,network_distr_args=network_distr_args)

    if verbose: # print uid and uids of contacts
        uids = contacts.keys()
        uids = [uid for uid in uids]
        for uid in uids:
            print(uid, contacts[uid]['contacts']['M'])


    return contacts


def make_contacts_by_social_layers_and_age_mixing(n=1e4,location='seattle_metro',state_location='Washington',country_location='usa',sheet_name='United States of America',verbose=False):
    """
    Make a popdict of n people, age and sex sampled from Seattle Metro demographics.
    Contacts are created and stored by layer. 

    Layers are : 
        'H' - households
        'S' - schools
        'W' - workplaces
        'C' - general community

    Use sheet_name to decide which set of age mixing patterns to sample contact 
    ages from. Age mixing patterns in the population will match this approximately, 
    but contact networks will still be random in the sense that clustering or 
    triangles will not be enforced. For example, an individual's contacts in the 
    household layer ('H') will match what's expected given their age, but their 
    contacts won't have the same contacts as them. This means the model is not 
    directly creating households, schools, or worksplaces, but contacts for each
    individual similar to those expected in terms of age. Caveat: students/teachers 
    interact at school and workers interact at work, but they won't interact in both.

    What's the purpose of this without clustering you ask? 

    Compartmental models routinely use age mixing matrices to model the effects
    of age mixing patterns on infectious disease spreading at the aggregate level.
    Agent based models require information at the individual level and this allows us
    to bring some of that age mixing from compartmental models into an agent based 
    modeling framework. 

    """
    popdict = sp.make_popdict(n=n,location=location,state_location=state_location,country_location=country_location,use_demography=True)

    n_contacts_dic = {'H': 7, 'S': 20, 'W': 20, 'C': 10} # dict of the average number of contacts per layer
    options_args = {'use_age_mixing': True, 'use_social_layers': True}

    contacts = sp.make_contacts(popdict,n_contacts_dic=n_contacts_dic,location=location,state_location=state_location,country_location=country_location,sheet_name=sheet_name,options_args=options_args)
    
    if verbose:
        layers = ['H','S','W','C']
        uids = contacts.keys()
        uids = [uid for uid in uids]
        for uid in uids:
            print(uid)
            for k in layers:
                print(k,contacts[uid]['contacts'][k])

    return contacts



if __name__ == '__main__':
    
    # point datadir where your data folder lives
    datadir = sp.datadir

    # location information - currently we only support the Seattle Metro area in full, however other locations can be supported with this framework at a later date
    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    popdict = make_popdict_of_random_people()
    popdict = make_popdict_of_people()
    popdict = make_popdict_with_supplied_ages(datadir)

    contacts = make_random_contacts(average_degree=4) # average degree of 4
    contacts = make_contacts_by_social_layers_and_age_mixing(verbose=True)


