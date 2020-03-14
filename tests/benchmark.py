# Benchmark the simulation

import sciris as sc
import synthpops as sp

to_profile = 'make_contacts' # Must be one of the options listed below

func_options = {'make_popdict': sp.make_popdict,
                'make_contacts': sp.make_contacts,
                }

def make_contacts():

    # Copied from test_contacts.py
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79}
    weights_dic['R'] = 7 # increase the general community weight because the calibrate weight 2.79 doesn't include contacts from the general community that you don't know but are near!
    n = 10000

    kwargs = dict(weights_dic=weights_dic,
                  use_social_layers=True,
                  directed=False,
                  use_student_weights=True)

    popdict = sp.make_popdict(n=n)
    contacts = sp.make_contacts(popdict, **kwargs)

    return contacts

sc.profile(run=make_contacts, follow=func_options[to_profile])


