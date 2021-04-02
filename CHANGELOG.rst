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

- "Github Information": the associated PRs to any changes.


~~~~~~~~~~~~~~~~~~~~~~~
Latest versions (1.6.x)
~~~~~~~~~~~~~~~~~~~~~~~


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



