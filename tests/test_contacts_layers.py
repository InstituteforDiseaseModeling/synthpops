import synthpops as sp
import unittest
import json
import os
from copy import deepcopy

#region pre-test setup
datadir = sp.datadir  # point datadir where your data folder lives

# location information - currently we only support the Seattle Metro area in full, however other locations can be supported with this framework at a later date
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'
sheet_name = 'United States of America'
level = 'county'

n = 20000
verbose = True
plot = True

# load population into a dictionary of individuals who know who their contacts are
options_args = {'use_microstructure': True}
network_distr_args = {'Npop': n}
contacts = sp.make_contacts(location=location, state_location=state_location,
                            country_location=country_location, options_args=options_args,
                            network_distr_args=network_distr_args)

# close_contacts_number = {'S': 10, 'W': 10}
# CONTACTS = sp.trim_contacts(contacts, trimmed_size_dic=close_contacts_number)
CONTACTS = contacts
#endregion


class SynthpopsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contacts = CONTACTS
        self.is_debugging = False
        pass

    def tearDown(self) -> None:
        pass

    def find_fresh_uid(self, uid_list,
                       last_index_checked,
                       checked_people):
        my_uid = None
        while not my_uid:
            potential_uid = uid_list[last_index_checked]
            if potential_uid not in checked_people:
                my_uid = potential_uid
                checked_people.append(my_uid)
                pass
            else:
                last_index_checked += 1
            pass
        return my_uid, last_index_checked, checked_people

    def make_person_json(self, popdict, target_uid):
        """
        creates a dictionary with root members 'uid', 'age', 'sex', 'loc', 'contacts'
        contacts is itself a dictionary of layers, each key has a list of uids of people.
        """
        my_person = popdict[target_uid]

        person_json = {'uid': target_uid}
        for k in ['age', 'sex', 'loc']:
            person_json[k] = my_person[k]
            pass
        contact_keys = my_person['contacts'].keys()

        person_json['contacts'] = {}
        for k in list(contact_keys):
            these_contacts = my_person['contacts'][k]
            uids = []
            for uid in these_contacts:
                uids.append(uid)
                pass
            person_json['contacts'][k] = uids
            pass
        return person_json

    def check_bidirectionality_of_contacts(self, person_json, popdict):
        my_uid = person_json['uid']
        for k in person_json['contacts']:
                expected_uids = person_json['contacts'][k]
                for uid in expected_uids:
                    friend = popdict[uid]
                    friend_contact_group = list(friend['contacts'][k])
                    self.assertIn(my_uid, friend_contact_group)
                    pass
        pass

    def test_contacts_are_bidirectional(self):
        num_people = 5
        checked_people = []
        last_index_checked = 0
        self.is_debugging = False
        while len(checked_people) < num_people:
            popdict = deepcopy(self.contacts)

            uids = popdict.keys()
            uid_list = list(uids)
            my_uid, last_index_checked, checked_people = \
                self.find_fresh_uid(uid_list=uid_list,
                                    last_index_checked=last_index_checked,
                                    checked_people=checked_people)

            person_json = self.make_person_json(popdict=popdict,
                                                target_uid=my_uid)

            if self.is_debugging:
                person_filename = f"DEBUG_popdict_person_{my_uid}.json"
                print(f"TEST: {my_uid}")
                if os.path.isfile(person_filename):
                    os.unlink(person_filename)
                    pass
                with open(person_filename,"w") as outfile:
                    json.dump(person_json, outfile, indent=4, sort_keys=True)
                    pass
                pass

            # Now check that each person in each network has me in their network
            self.check_bidirectionality_of_contacts(person_json=person_json,
                                                    popdict=popdict)

    def test_contact_layers_are_same_for_all_members(self):
        # Get four persons, one each with a home, work, school, and community layer
        popdict = deepcopy(self.contacts)
        representative_people = {
            "H": None,
            "S": None,
            "W": None
        }
        # TODO: add "C": None to above dictionary if that is ever supported here.
        try_this_next = 0
        indexes = list(popdict.keys())
        for k in representative_people.keys():
            # Loop through, and find at least one person with each layer.
            while not representative_people[k]:
                temp_person = popdict[indexes[try_this_next]]
                if try_this_next % 1000 == 0:
                    print(f"At index {try_this_next} of {len(popdict)}")
                if len(temp_person['contacts'][k]) > 0:
                    print(f"Found my {k}\n")
                    representative_people[k] = indexes[try_this_next]
                elif try_this_next < 1000:
                    try_this_next += 1
                else:  # We went through 1000 people and are missing one
                    break

        if self.is_debugging:
            print(f"Representative people: {representative_people}")
            print(f"Try this next: {try_this_next}")
            pass

        for k in representative_people:
            if representative_people[k]:
                # No one is in their own contact list. So first get a friend's layer
                layer_friends = list(popdict[representative_people[k]]['contacts'][k])
                other_layer_person = popdict[layer_friends[0]]
                layer_friends.remove(layer_friends[0]) # then add that person's uid back
                other_layer_friends = list(other_layer_person['contacts'][k])
                other_layer_friends.remove(representative_people[k])
                self.assertEqual(sorted(layer_friends), sorted(other_layer_friends),
                                 msg="The lists of uids should be identical")
            else:
                print(f"This is totally embarassing, no one found with layer {k}\n")

        pass

    def test_trimmed_contacts_are_bidirectional(self):
        num_people = 5
        checked_people = []
        last_index_checked = 0
        close_contacts_numbers = {'S': 10, 'W': 10}
        self.is_debugging = False

        my_contacts = sp.make_contacts(location=location, state_location=state_location,
                                      country_location=country_location, options_args=options_args,
                                      network_distr_args=network_distr_args)
        my_trim_contacts = sp.trim_contacts(my_contacts, trimmed_size_dic=close_contacts_numbers)

        popdict = my_trim_contacts

        uids = popdict.keys()
        uid_list = list(uids)


        while len(checked_people) < num_people:
            my_uid, last_index_checked, checked_people = \
                self.find_fresh_uid(uid_list=uid_list,
                                    last_index_checked=last_index_checked,
                                    checked_people=checked_people)

            first_person = popdict[my_uid]

            person_json = self.make_person_json(popdict=popdict,
                                                target_uid=my_uid)

            if self.is_debugging:
                person_filename = f"DEBUG_popdict_person_{my_uid}.json"
                print(f"TEST: {my_uid}")
                if os.path.isfile(person_filename):
                    os.unlink(person_filename)
                    pass
                with open(person_filename,"w") as outfile:
                    json.dump(person_json, outfile, indent=4, sort_keys=True)
                    pass
                pass

            # Now check that each person in each network has me in their network
            self.check_bidirectionality_of_contacts(person_json=person_json,
                                                    popdict=popdict)
        pass




