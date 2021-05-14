"""
Test sp.location_data properties
"""
import pytest
import tempfile
import types
import os
import synthpops as sp

log = sp.logger
properties = {k:v for k,v in vars(sp.data.Location).items() if not k.startswith('_') and not isinstance(v, types.FunctionType)}
# these properties are missing due to no data
skipped_properties = {"all":
                          ["data_provenance_notices",
                           "reference_links",
                           "citations",
                           "notes",
                           "ltcf_num_staff_distribution",
                           "ltcf_num_residents_distribution"],
                      "Oregon":
                          ["household_head_age_brackets",
                           "ltcf_resident_to_staff_ratio_distribution",
                           "ltcf_use_rate_distribution",
                           "school_size_brackets",
                           "school_size_distribution",
                           "school_size_distribution_by_type",
                           "school_types_by_age",
                           ],
                      "Senegal":
                          ["population_age_distribution_20",
                           "ltcf_resident_to_staff_ratio_distribution",
                           "ltcf_use_rate_distribution",
                           "school_size_distribution_by_type",
                           "school_types_by_age",
                           ]
                      }
locations = \
    {'usa':
        [
            {'Washington': ['seattle_metro', 'Spokane_County', 'Island_County', 'Franklin_County']},
            {'Oregon': ['portland_metro']}
        ],
    'Senegal': [{'Dakar': ['Dakar']}]
}

test_tuples =[]
# build test tuples from locations
for i in locations:
    log.debug(f"---------- Country: {i} ----------")
    country_location = i
    for j in locations[i]:
        log.debug(f"........... State: {list(j.keys())[0]} ..........")
        state_location = list(j.keys())[0]
        for k in j[state_location]:
            log.debug(f"*********** Location {k} **********")
            specific_location = k
            test_tuples.append((country_location, state_location, specific_location))


@pytest.mark.parametrize(('country_location', 'state_location', 'specific_location'), test_tuples)
def test_location_data(specific_location, state_location, country_location):
    location_data = sp.load_location(specific_location, state_location, country_location, revert_to_default=False)
    skiplist = set(skipped_properties["all"])
    for s in skipped_properties:
        if s in location_data.location_name:
            skiplist = skiplist.union(set(skipped_properties[s]))
    log.debug(test_tuples)
    for p in properties:
        if p not in skiplist:
            log.debug("===============")
            log.debug(f"property:{p}")
            value = location_data.__getattribute__(p)
            log.debug(f"value:{value}")
            if len(value) < 1:
                log.debug(f"property:{p} length 0")
            assert len(value) > 0, f"property:{p} length 0 for {test_tuples}"


def test_location_default():
    """
    if location is not available, seattle_metro should be used
    """
    specific_location = "Sacramento"
    state_location = "California"
    country_location = "usa"
    location_data = sp.load_location(specific_location, state_location, country_location, revert_to_default=True)
    assert "seattle_metro" in location_data.location_name
    assert "Washington" in location_data.location_name
    assert "usa" in location_data.location_name


def test_parent_data_loaded():
    """
    test seattle_metro
    household_head_age_brackets
    household_head_age_distribution_by_family_size
    should be loaded from the Washington level
    """
    specific_location = "seattle_metro"
    state_location = "Washington"
    country_location = "usa"
    child_location = sp.load_location(specific_location, state_location, country_location, revert_to_default=False)
    parent_location = sp.load_location(specific_location=None,
                                       state_location=state_location,
                                       country_location=country_location, revert_to_default=True)
    log.debug(f"parent_location: {parent_location.location_name}")
    log.debug(f"child_location: {child_location.location_name}")
    # both should have 11 age brackets and 8 family sizes
    assert len(parent_location.household_head_age_brackets) == 11, \
        "household_head_age_brackets incorrect for Washington."
    assert len(parent_location.household_head_age_distribution_by_family_size) == 8, \
        "household_head_age_distribution_by_family_size incorrect for Washington."
    assert parent_location.household_head_age_brackets == child_location.household_head_age_brackets, \
        "household_head_age_brackets incorrect for seattle_metro."
    assert parent_location.household_head_age_distribution_by_family_size == child_location.household_head_age_distribution_by_family_size, \
        "household_head_age_distribution_by_family_size incorrect for seattle_metro."


def test_brackets_unavailable():
    """
    Test get_population_age_distribution
    Dakar has only 16/18 age brackets
    """
    specific_location = "Dakar"
    state_location = "Dakar"
    country_location = "Senegal"
    location_data = sp.load_location(specific_location, state_location, country_location, revert_to_default=False)
    assert len(location_data.get_population_age_distribution(nbrackets=16)) == 16
    assert len(location_data.get_population_age_distribution(nbrackets=18)) == 18

    specific_location = "portland_metro"
    state_location = "Oregon"
    country_location = "usa"
    location_data = sp.load_location(specific_location, state_location, country_location, revert_to_default=False)
    assert len(location_data.get_population_age_distribution(nbrackets=16)) == 16
    assert len(location_data.get_population_age_distribution(nbrackets=18)) == 18
    assert len(location_data.get_population_age_distribution(nbrackets=20)) == 20
    with pytest.raises(RuntimeError) as err:
        location_data.get_population_age_distribution(nbrackets=21)
        assert "Unsupported value for nbrackets" in str(err.value)


def test_attribute_unavailable():
    location_data = sp.load_location(None, None, None, revert_to_default=True)
    with pytest.raises(AttributeError):
        location_data.school_attr_noexistent


def test_save_location_to_filepath():
    location_data = sp.load_location(None, None, None, revert_to_default=True)
    outfile = tempfile.mktemp(suffix=".json")
    try:
        sp.save_location_to_filepath(location_data, outfile)
        assert os.stat(outfile).st_size > 0
        saved_location_data = sp.load_location_from_json_str(open(outfile, 'r').read())
        assert saved_location_data.location_name == location_data.location_name
    finally:
        os.remove(outfile) if os.path.exists(outfile) else None


def test_populate_parent_exception():
    """
    Test if loading parent data from a wrong file, the same location is returned
    if populate_parent_data is not a vaild json string or json type, error is returned
    """
    specific_location = ""
    state_location = ""
    country_location = "usa"
    location_data = sp.load_location(specific_location, state_location, country_location, revert_to_default=False)
    parent_data = sp.data.populate_parent_data_from_file_path(location_data, "badfile.json")
    assert parent_data == location_data
    # pass parent in list which is not supported
    with pytest.raises(RuntimeError) as err:
        test_location = sp.load_location_from_json_str('{"location_name": "X"}')
        test_location.parent = ["Y"]
        sp.populate_parent_data(test_location)
        assert "Invalid type" in str(err)


if __name__ == "__main__":
    testcase = 'test_location_data'
    pytest.main(['-v', '-k', testcase])
