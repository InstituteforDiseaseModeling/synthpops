import synthpops as sp

def test_load_location():
    test_str = """{
  "data_provenance_notices": ["notice1","notice2"],
  "reference_links": ["reference1","reference2"],
  "citations": ["citation1","citation2"],
  "population_age_distribution_brackets": [
    [0,4,0.06],
    [5,9,0.20]
  ]
}"""
    location = sp.load_location_from_json_str(test_str)

    assert len(location.data_provenance_notices) == 2
    assert location.data_provenance_notices[0] == "notice1"
    assert location.data_provenance_notices[1] == "notice2"

    assert len(location.reference_links) == 2
    assert location.reference_links[0] == "reference1"
    assert location.reference_links[1] == "reference2"

    assert len(location.citations) == 2
    assert location.citations[0] == "citation1"
    assert location.citations[1] == "citation2"

    assert len(location.population_age_distribution_brackets) == 2
    assert len(location.population_age_distribution_brackets[0]) == 3
    assert location.population_age_distribution_brackets[0][0] == 0
    assert location.population_age_distribution_brackets[0][1] == 4
    assert location.population_age_distribution_brackets[0][2] == 0.06

    assert len(location.population_age_distribution_brackets[1]) == 3
    assert location.population_age_distribution_brackets[1][0] == 5
    assert location.population_age_distribution_brackets[1][1] == 9
    assert location.population_age_distribution_brackets[1][2] == 0.20

