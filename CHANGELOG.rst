==========
What's new
==========

Starting with SynthPops version 1.5.2, this file will document all changes to the codebase. By nature, SynthPops is a library to generate stochastic networked populations, so over time there will be model and code updates that change regression results. When these kinds of changes are made, we'll flag that here with the term "Regression Information". In addition, here are some other terms useful for understanding updates documented here.


~~~~~~~~~~~~~~~~~~~~
Legend for changelog
~~~~~~~~~~~~~~~~~~~~

- "Feature": a new feature previously unavailable.

- "Efficiency": a refactor of a previous method to make the calculation faster or require less memory.

- "Fix": a fix to a bug in the code base where a method either did not work under certain conditions or results were not as expected.

- "Deprecated": a method or feature that has been removed or support will be removed for in the future.

- "Regression Information": a change to the model or update to data resulted in a change to regression results.

- "Github Info": the associated PRs to any changes.


~~~~~~~~~~~~~~~~~~~~~~~
Latest versions (1.7.x)
~~~~~~~~~~~~~~~~~~~~~~~


Version 1.7.3 (2021-04-14)
--------------------------
- Data folder cleaned up and removed individual csv data files now that synthpops has json data files instead for the collection of data used for each location.
- Json data objects also updated with documentation on the sources for the original and estimated data. When data have been estimated or inferred, to the best of our ability, we've added a note about this in the notes field.
- *Github Info*: PR `427 <https://github.com/amath-idm/synthpops/pull/427>`__


Version 1.7.2 (2021-04-13)
--------------------------
- *Feature*: Re-enabled support of age distributions for any number of age brackets. Json data files have been updated to accomodate this flexibility.
- *Github Info*: PR `422 <https://github.com/amath-idm/synthpops/pull/422>`__


Version 1.7.1 (2021-04-09)
--------------------------
- Feature: Added checks for probability distributions with methods ``sp.check_all_probability_distribution_sums()``, ``sp.check_all_probability_distrubution_nonnegative()``, ``sp.check_probability_distribution_sum()``, ``sp.check_probability_distribution_nonnegative()``. These check that probabilities sum to 1 within a tolerance level  (0.05), and have all non negative values. Added method to convert data from pandas dataframe to json array style, ``sp.convert_df_to_json_array()``. Added statistical test method ``sp.statistic_test()``. Added method to count contacts, ``sp.get_contact_counts_by_layer()``, and method to plot the results, ``sp.plot_contact_counts()``. See ``sp.contact_networks.get_contact_counts_by_layer()`` for more details on the method.
- Added example of how to load data into the location json objects and save to file. See ``examples/create_location_data.py`` and ``examples/modify_location_data.py``.
- *Github Info*: PR `410 <https://github.com/amath-idm/synthpops/pull/410>`__, `413 <https://github.com/amath-idm/synthpops/pull/413>`__, `423 <https://github.com/amath-idm/synthpops/pull/423>`__


Version 1.7.0 (2021-04-05)
--------------------------
- *Efficiency*: Major refactor of data methods to read from consolidated json data files for each location and look for missing data from parent locations or alternatively json data files for default locations. Migration of multiple data files for locations into a single json object per location under the ``data`` directory. This will should make it easier to identify all of the available data per location and where missing data are read in from. Examples of how to create, change, and save new json data files will come in the next minor version update.
- *Feature*: Location data jsons now have fields for the data source, reference links, and citations! These fields will be fully populated shortly. Please reference the links provided for any data obtained from SynthPops as most population data are sourced from other databases and should be referenced as such.
- *Deprecated*: Refactored data methods no longer support the reading in of data from user specified file paths. Use of methods to read in age distributions aggregated to a number of age brackets not equal to 16, 18, or 20 (officially supported values) is currently turned off. Next minor update will re-enable these features. Old methods are available in `synthpops.data_distributions_legacy.py`, however this file will be removed in upcoming versions once we have migrated all examples to use the new data methods and have fully enabled all the functionality of the original data methods. Please update your usage of SynthPops accordingly.
- Updated documentation about the input data layers.
- *Github Info*: PR `407 <https://github.com/amath-idm/synthpops/pull/407>`__, `303 <https://github.com/amath-idm/synthpops/pull/303>`__


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.6.x (1.6.0 – 1.6.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.6.2 (2021-04-01)
--------------------------
- *Feature*: Added new methods, ``sp.get_household_head_ages_by_size()``, ``sp.plot_household_head_ages_by_size()``. Also accessible pop methods as ``pop.get_household_head_ages_by_size()``, ``pop.plot_household_head_ages_by_size()``. These calculate the generated count the household head age by the household size, and the plotting methods compare this to the expected age distributions by size as matrices.
- *Github Info*: PR `385 <https://github.com/amath-idm/synthpops/pull/385>`__


Version 1.6.1 (2021-03-25)
--------------------------
- *Feature*: Added new methods, ``sp.check_dist()`` and aliases ``sp.check_normal()`` and ``sp.check_poisson()``, to check whether the observed distribution matches the expected distribution.
- *Github Info*: PR `373 <https://github.com/amath-idm/synthpops/pull/373>`__


Version 1.6.0 (2021-03-20)
--------------------------
- *Feature*: Adding summary methods for SynthPops pop objects accesible as pop.summary and computed using pop.compute_summary(). Also adding several plotting methods for these summary data.
- Updating ``sp.workplaces.assign_rest_of_workers()`` to work off a copy of the workplace age mixing matrix so that the copy stored in SynthPops pop objects is not modified during generation.
- More tests for summary methods in pop.py, methods in config.py, plotting methods in plotting.py
- *Regression Information*: Adding new workplace size data specific for the Seattle metro area which changes the regression results. The previous data from the Washington state level and the new data for the metropolitan statistical area (MSA) of Seattle for the 2019 year are very similar, however the use of this data with random number generators does result in slight stochastic differences in the populations generated. 
- *Github Info*: PRs `356 <https://github.com/amath-idm/synthpops/pull/356>`__, `357 <https://github/com/amath-idm/synthpops/pull/357>`__, `358 <https://github.com/amath-idm/synthpops/pull/358>`__, `360 <https://github.com/amath-idm/synthpops/pull/360>`__



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Versions 1.5.x (1.5.2 – 1.5.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Version 1.5.3 (2021-03-16)
--------------------------
- *Deprecated*: Removing use of verbose parameter to print statements to use logger.debug() instead and removing the verbose parameter where deprecated.
- *Github Info*: PRs `363 <https://github.com/amath-idm/synthpops/pull/363>`__, `379 <https://github.com/amath-idm/synthpops/pull/379>`__, `380 <https://github.com/amath-idm/synthpops/pull/380>`__


Version 1.5.2 (2021-03-09)
--------------------------
- *Feature*: Added metadata to pop objects.
- Updated installation instructions and reference citation.
- *Github Info*: PRs `365 <https://github.com/amath-idm/synthpops/pull/365>`__, `351 <https://github.com/amath-idm/synthpops/pull/351>`__



