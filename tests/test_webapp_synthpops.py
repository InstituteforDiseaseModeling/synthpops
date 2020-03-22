import numpy as np
import synthpops as sp
import sciris as sc


def test_webapp_synthpops_calls(n,location='seattle_metro',state_location='Washington',use_bayesian=True,sheet_name='United States of America'):
    datadir = sp.datadir

    sp.read_age_bracket_distr(datadir,location,state_location=state_location,use_bayesian=use_bayesian)
    sp.get_census_age_brackets(datadir,use_bayesian=use_bayesian)
    num_agebrackets = 16

    n_contacts_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 7}

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir,location,num_agebrackets,use_bayesian,sheet_name)
    contact_matrix_dic['M'] = sp.combine_matrices(contact_matrix_dic,n_contacts_dic,num_agebrackets)

    for k in contact_matrix_dic:
        print(contact_matrix_dic[k].shape)

    n = int(n)
    sp.get_age_n(n,location=location,state_location=state_location,use_bayesian=use_bayesian)

    return


def test_webapp_contacts_calls(n,location='seattle_metro',state_location='Washington',use_bayesian=True,sheet_name='United States of America'):

    n = int(n)
    popdict = sp.make_popdict(n=n,state_location=state_location,location=location,use_bayesian=True)

    print(popdict)

    return


def test_webapp_make_contacts(n,n_contacts_dic=None,state_location='Washington',location='seattle_metro',use_bayesian=True,sheet_name='Pakistan'):
    sc.heading(f'Making popdict for {n} peoplse')
    n = int(n)
    popdict = sp.make_popdict(n=n,state_location=state_location,location=location,use_bayesian=use_bayesian)

    options_args = dict.fromkeys(['use_age','use_bayesian'], True)
    contacts = sp.make_contacts(popdict,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location,sheet_name=sheet_name,options_args=options_args)

    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n,uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print(n,'uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,len(contact_ages),'contact ages',contact_ages)
        print()

    return popdict


def test_webapp_make_contacts_and_show_some_layers(n,n_contacts_dic=None,state_location='Washington',location='seattle_metro',use_bayesian=True,sheet_name='United States of America'):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)
    popdict = sp.make_popdict(n=n,state_location=state_location,location=location,use_bayesian=use_bayesian)

    options_args = dict.fromkeys(['use_age','use_social_layers','use_bayesian'], True)
    contacts = sp.make_contacts(popdict,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location,sheet_name=sheet_name,options_args=options_args)

    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n,uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print(n,'uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,len(contact_ages),'contact ages',contact_ages)
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


    test_webapp_synthpops_calls(n,location,state_location,use_bayesian,sheet_name)
    # test_webapp_contacts_calls(n,location,state_location,use_bayesian,sheet_name)
    test_webapp_make_contacts(n,state_location=state_location,location=location)
    test_webapp_make_contacts_and_show_some_layers(n,state_location=state_location,location=location)