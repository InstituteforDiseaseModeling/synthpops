import numpy as np
import synthpops as sp
import sciris as sc

default_n = 1000


def test_webapp_synthpops_calls(n=default_n, location='seattle_metro', state_location='Washington',
                                country_location='usa', sheet_name='United States of America'):
    datadir = sp.datadir

    sp.read_age_bracket_distr(datadir, location=location, state_location=state_location,
                              country_location=country_location)
    sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    num_agebrackets = 16

    n_contacts_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'C': 7}

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    contact_matrix_dic['M'] = sp.combine_matrices(contact_matrix_dic, n_contacts_dic, num_agebrackets)

    for k in contact_matrix_dic:
        print(contact_matrix_dic[k].shape)

    n = int(n)
    sp.get_age_n(datadir, n=default_n, location=location, state_location=state_location,
                 country_location=country_location)

    return


def test_webapp_contacts_calls(n=default_n, location='seattle_metro', state_location='Washington',
                               country_location='usa'):
    n = int(n)
    popdict = sp.make_popdict(n=n, state_location=state_location, location=location, country_location=country_location)

    print(popdict)

    return


def test_webapp_make_contacts(n=default_n, n_contacts_dic=None, state_location='Washington', location='seattle_metro',
                              country_location='usa', sheet_name='Pakistan'):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)
    popdict = sp.make_popdict(n=n, state_location=state_location, location=location, country_location=country_location)

    options_args = dict.fromkeys(['use_age', 'use_age_mixing'], True)
    contacts = sp.make_contacts(popdict, n_contacts_dic=n_contacts_dic, state_location=state_location,
                                location=location, country_location=country_location, sheet_name=sheet_name,
                                options_args=options_args)

    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n, uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print(n, 'uid', uid, 'age', contacts[uid]['age'], 'total contacts',
              np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k, len(contact_ages), 'contact ages', contact_ages)
        print()

    return popdict


def test_webapp_make_contacts_and_show_some_layers(n=default_n, n_contacts_dic=None, state_location='Washington',
                                                   location='seattle_metro', country_location='usa',
                                                   sheet_name='United States of America'):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)
    popdict = sp.make_popdict(n=n, state_location=state_location, location=location, country_location=country_location)

    options_args = dict.fromkeys(['use_age', 'use_social_layers', 'use_age_mixing'], True)
    contacts = sp.make_contacts(popdict, n_contacts_dic=n_contacts_dic, state_location=state_location,
                                location=location, country_location=country_location, sheet_name=sheet_name,
                                options_args=options_args)

    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n, uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print(n, 'uid', uid, 'age', contacts[uid]['age'], 'total contacts',
              np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k, len(contact_ages), 'contact ages', contact_ages)
        print()

    return popdict


if __name__ == '__main__':
    sp.validate()
    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'

    # location = 'portland_metro'
    # state_location = 'Oregon'
    country_location = 'usa'

    use_bayesian = True

    # sheet_name = 'United States of America'
    sheet_name = 'Algeria'

    n = int(3e3)

    test_webapp_synthpops_calls(n, location=location, state_location=state_location, country_location=country_location,
                                sheet_name=sheet_name)
    test_webapp_contacts_calls(n, location=location, state_location=state_location, country_location=country_location,
                               sheet_name=sheet_name)
    test_webapp_make_contacts(n, state_location=state_location, location=location, country_location=country_location)
    test_webapp_make_contacts_and_show_some_layers(n, state_location=state_location, location=location,
                                                   country_location=country_location)
