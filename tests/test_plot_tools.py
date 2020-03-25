import synthpops as sp
import sciris as sc
import numpy as np
import math
import copy
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
import functools
import os


font_path = sp.datadir.replace('synthpops','GoogleFonts')
print(font_path)

datadir = sp.datadir

def test_make_contacts_with_layers(n,n_contacts_dic=None,state_location='Oregon',location='portland_metro',use_usa=True,use_bayesian=False):
    n = int(n)
    popdict = sp.make_popdict(n=n,state_location=state_location,location=location,use_usa=use_usa,use_bayesian=use_bayesian)

    options_args = dict.fromkeys(['use_age','use_sex','use_loc','use_social_layers','use_activity_rates'],True)
    options_args['use_usa'] = use_usa
    options_args['use_bayesian'] = use_bayesian
    contacts = sp.make_contacts(popdict,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location,options_args=options_args)

    return contacts


def calculate_contact_frequency(contacts):
    uids = contacts.keys()
    uids = [uid for uid in uids]

    num_ages = 100

    F_dic = {}
    for k in ['M','H','S','W','R']:
        F_dic[k] = np.zeros((num_ages,num_ages))

    for n,uid in enumerate(uids):
        if n > 2:
            break
        layers = contacts[uid]['contacts']
        age = contacts[uid]['age']
        # print(n,'uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            F_dic[k][age, contact_ages] += 1

    return F_dic


def plot_contact_frequency():
    return 0


if __name__ == '__main__':
    
    n = int(2e3)
    n_contacts_dic = {'H': 3, 'S': 20, 'W': 20, 'R': 10}

    state_location = 'Washington'
    location = 'seattle_metro'

    contacts = test_make_contacts_with_layers(n,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location)

    print(len(contacts))

    F_dic = calculate_contact_frequency(contacts)
    print(F_dic)

    sp.say_hi()