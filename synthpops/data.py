import json
from jsonobject import *


class Location(JsonObject):
    data_provenance_notices = ListProperty(StringProperty)
    reference_links = ListProperty(StringProperty)
    citations = ListProperty(StringProperty)
    population_age_distribution_brackets = ListProperty(ListProperty(FloatProperty))


def load_location_from_json(json_obj):
    location = Location(json_obj)
    return location


def load_location_from_json_str(json_str):
    json_obj = json.loads(json_str)
    return load_location_from_json(json_obj)


def load_location_from_filepath(filepath):
    f = open(filepath, 'r')
    json_obj = json.load(f)
    return load_location_from_json(json_obj)

