import synthpops.data_distributions_legacy as data_distributions_legacy
import synthpops.data as data
import synthpops.config as spconfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default=None, required=False, help='Source data directory. If unspecified, uses whatever synthpops default is.')
parser.add_argument("--country_location", type=str, default=None, required=True, help='Input country.')
parser.add_argument("--state_location", type=str, default=None, required=False, help='Input state.')
parser.add_argument("--location", type=str, default=None, required=False, help='Input location.')
parser.add_argument("--output_folder", type=str, default=None, required=False, help="Output folder. Default is working directory.")
args = parser.parse_args()

def migrate_legacy_data(datadir, country_location, state_location, location, output_folder):
    new_location = data.Location()
    parent = ""
    if country_location is None:
        raise RuntimeError(f"country_location must be provided")
    if location is None and state_location is None:
        new_location.location_name = country_location
        parent = ""
    elif location is None:
        new_location.location_name = f"{country_location}-{state_location}"
        parent = country_location
    else:
        new_location.location_name = f"{country_location}-{state_location}-{location}"
        parent = f"{country_location}-{state_location}"

    new_location.parent = parent

    legacy_age_brackets = data_distributions_legacy.get_census_age_brackets(datadir,
                                                                            location,
                                                                            state_location,
                                                                            country_location
                                                                            )
    for k, age_range in legacy_age_brackets.items():
        # Insert age bracket with 0.0 percentage for now.
        new_location.population_age_distribution.append([float(min(age_range)), float(max(age_range)), 0.0])


    legacy_age_distribution = data_distributions_legacy.read_age_bracket_distr(datadir,
                                                                               location,
                                                                               state_location,
                                                                               country_location)

    for k, dist_percentage in legacy_age_distribution.items():
        new_location.population_age_distribution[k][2] = dist_percentage

    return


if __name__ == '__main__':

    datadir = args.datadir
    if datadir is None:
        datadir = spconfig.datadir

    migrate_legacy_data(datadir,
                        args.country_location,
                        args.state_location,
                        args.location,
                        args.output_folder)