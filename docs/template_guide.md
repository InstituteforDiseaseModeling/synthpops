# Quick guide to how to generate your own synthetic population

## Template

A template exists in `synthpops/examples/generate_uk_population.py`.

This template contains six functions:

* `get_cv_age_distribution(location)`: reads in age distribution data from `Covasim` and processes them into a standard format that `Synthpops` can work with

* `get_cv_age_brackets(location)`: reads in age brackets from `Covasim` and processes them into a standard format that `Synthpops` can work with

* `resample_age_uk(exp_age_distr, a)`: resamples age and gets used to interpolate United Kingdom age distribution data by age brackets into single year age distributions

* `generate_larger_households(size, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)`: generates households larger than size 1

* `generate_all_households(N, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)`: generates all households

* `generate_synthetic_population_uk(n, datadir, location='uk', state_location='uk', country_location='uk', sheet_name='United Kingdom of Great Britain', school_enrollment_counts_available=False, verbose=False, plot=False, write=False, use_default=True)`: generates synthetic population with micro structure for the United Kingdom


The idea behind this template is that users can either provide data for the location they are interested in, or if some data are not available, then default values from Seattle, Washington are used in place. The template is heavily annotated and should be sufficient to using it to model your synthetic populations with micro structure using `Synthpops`.

Users can supply their own data files to most data functions by either setting up data folders or supplying these functions with the file paths to each data file. In general, `Synthpops` data functions will look for data with the 
file path:

`./data/demographics/contact_matrices_152_countries/country_location/state_location/location/data_type/data_file`

Users can also supply the file paths to their own data by setting the parameter `file_path` to the file path. For data that are binned or given by brackets, users supplying their own distribution data must also supply files with those brackets. All bracket files take the same format, with each line listing the start and end of the bracket. For example, for school size brackets, the Seattle Metro area file looks like:

    0,50
    51,100
    101,300
    301,500
    501,700
    701,900
    901,1100
    1101,1300
    1301,1500
    1501,1700
    1701,1900
    1901,2100
    2101,2300
    2301,2700

Many places will have different school sizes that do not fit welll within these brackets, so users should consider making their own files for this that are more specific for their population. In particular, if the location under study has much larger schools, these brackets will not capture that.
