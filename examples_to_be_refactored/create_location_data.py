"""Example showing how to use the synthpops data api to create a new location file and save it."""

import synthpops as sp
import os

if __name__ == '__main__':
    # After we create the location data, we'll save it here.
    output_location_filepath = "example_created_location.json"

    print('Creating location data...')

    location_data = sp.Location()


    location_data.location_name = "Everythingbagelton"
    location_data.data_provenance_notices = ["This data was completely made up."]
    location_data.reference_links = ["n/a"]
    location_data.citations = ["n/a"]
    location_data.notes = ["This location is rumored to be the place that cream cheese was invented."]

    location_data.parent = None

    location_data.population_age_distributions = []
    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())

    location_data.population_age_distributions[0].num_bins = 2

    location_data.population_age_distributions[0].distribution = [
        [0,   5, 0.1],
        [6, 110, 0.9]
    ]

    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())
    location_data.population_age_distributions[1].num_bins = 4
    location_data.population_age_distributions[1].distribution = [
        [0,    5, 0.25],
        [6,   20, 0.25],
        [21,  60, 0.25],
        [61, 100, 0.25]
    ]



    location_data.employment_rates_by_age = [
        [0,  5,   0.0],
        [6,  25,  0.5],
        [26, 55,  0.5],
        [56, 110, 0.0]
    ]

    location_data.enrollment_rates_by_age = [
        [0,  55,   0.0],
        [56, 65,   0.1],
        [66, 110,  0.2]
    ]

    location_data.household_head_age_brackets = [
        [0, 25],
        [26, 50],
        [50, 110]
    ]

    location_data.household_head_age_distribution_by_family_size = [
        [1, 100.0, 200.0, 100.0],
        [2,  10.0,  20.0,  10.0],
        [3,  20.0,  10.0,  20.0],
        [4,   0.0,   0.0,   0.0]
    ]

    location_data.household_size_distribution = [
        [1, 0.25],
        [2, 0.25],
        [3, 0.25],
        [4, 0.25]
    ]

    location_data.ltcf_resident_to_staff_ratio_distribution = [
        [1, 5, 0.5],
        [6, 500, 0.5]
    ]

    location_data.ltcf_num_residents_distribution = [
        [1,  10, 0.25],
        [11, 20, 0.50],
        [21, 99, 0.25]
    ]

    location_data.ltcf_num_staff_distribution = [
        [1,  10, 0.50],
        [11, 20, 0.25],
        [21, 99, 0.25]
    ]

    location_data.ltcf_use_rate_distribution = [
        [65, 0.1],
        [66, 0.1],
        [67, 0.1],
        [68, 0.1],
        [69, 0.1],
        [70, 0.15],
        [71, 0.15],
        [72, 0.15]
    ]

    location_data.school_size_brackets = [
        [1,   100],
        [101, 500],
        [501, 5000]
    ]

    location_data.school_size_distribution = [
        0.25,
        0.50,
        0.25
    ]

    location_data.school_size_distribution_by_type = [
        sp.SchoolSizeDistributionByType(school_type="pk",
                                     size_distribution=[
                                         0.111, 0.111, 0.111
                                     ]),
        sp.SchoolSizeDistributionByType(school_type="es",
                                     size_distribution=[
                                         0.111, 0.111, 0.111
                                     ]),
        sp.SchoolSizeDistributionByType(school_type="ms",
                                     size_distribution=[
                                         0.111, 0.111, 0.111
                                     ])
    ]

    location_data.school_types_by_age = [
        sp.SchoolTypeByAge(school_type="pk",
                        age_range=[
                            1, 5
                        ]),
        sp.SchoolTypeByAge(school_type="es",
                        age_range=[
                            6, 10
                        ]),
        sp.SchoolTypeByAge(school_type="ms",
                        age_range=[
                            11, 13
                        ])
    ]

    location_data.workplace_size_counts_by_num_personnel = [
        [1,  10,  5000],
        [11, 20,  1000],
        [21, 100, 8000]
    ]

    print('... done.')

    # Save the location data.
    print(f'Saving location data to [{output_location_filepath}]')
    sp.save_location_to_filepath(location_data, output_location_filepath)

