'''
Test the new methods introduced in synthpops.people
'''

import numpy as np
import sciris as sc
import synthpops as sp
import pytest


def test_people():
    sc.heading('Basic People tests with plotting')
    pop = sp.Pop(n=50)
    ppl = pop.to_people()
    ppl.plot()
    ppl.plot_graph()
    return ppl


def test_advanced_people():
    sc.heading('Advanced People tests')

    # BasePeople methods
    ppl = sp.Pop(n=100).to_people()
    ppl.get(['age', 'sex'])
    ppl.keys()
    ppl.indices()
    ppl._resize_arrays(new_size=200) # This only resizes the arrays, not actually create new people
    ppl._resize_arrays(new_size=100) # Change back
    ppl.to_df()
    ppl.to_arr()
    ppl.person(50)
    people = ppl.to_people()
    ppl.from_people(people)
    ppl.make_edgelist([{'new_key':[0,1,2]}])
    ppl.brief()

    # Test adding populations
    p1 = sp.Pop(n=50).to_people()
    p2 = sp.Pop(n=100).to_people()
    p2.validate()
    p3 = p1 + p2
    assert len(p3) == len(p1) + len(p2)

    # Contacts methods
    contacts = ppl.contacts
    df = contacts['h'].to_df()
    ppl.remove_duplicates(df)
    with pytest.raises(sc.KeyNotFoundError):
        contacts['invalid_key']
    contacts.values()
    len(contacts)
    print(contacts)
    print(contacts['h'])

    # Layer methods
    hospitals_layer = sp.people.Layer()
    contacts.add_layer(hospitals=hospitals_layer)
    contacts.pop_layer('hospitals')
    df = hospitals_layer.to_df()
    hospitals_layer.from_df(df)

    # Generate an average of 10 contacts for 1000 people
    n = 10_000
    n_people = 1000
    p1 = np.random.randint(n_people, size=n)
    p2 = np.random.randint(n_people, size=n)
    beta = np.ones(n)
    layer = sp.people.Layer(p1=p1, p2=p2, beta=beta)

    # Convert one layer to another with extra columns
    index = np.arange(n)
    self_conn = p1 == p2
    layer2 = sp.people.Layer(**layer, index=index, self_conn=self_conn)
    assert len(layer2) == n
    assert len(layer2.keys()) == 5

    return ppl


def test_randpop():
    ppl = sp.people.make_people(n=100, pop_type='hybrid')
    return ppl


def test_age_structure():
    sc.heading('Age structures')

    available     = 'Lithuania'
    not_available = 'Ruritania'

    d = sc.objdict()
    d.age_data = sp.people.loaders.get_age_distribution(available)
    d.hh_data  = sp.people.loaders.get_household_size(available)

    with pytest.raises(ValueError):
        sp.people.get_age_distribution(not_available)

    return d


def test_other():
    sc.heading('Other People tests')

    # Test locations
    for location in [None, 'viet-nam']:
        sp.people.show_locations(location)
    return


if __name__ == '__main__':
    ppl  = test_people()
    ppl2 = test_advanced_people()
    ppl3 = test_randpop()
    data = test_age_structure()
    test_other()