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
Latest versions (1.5.x)
~~~~~~~~~~~~~~~~~~~~~~~


Version 1.5.3 (2021-03-16)
--------------------------
- Changing many of the verbose statements to use logger.debug() instead and removing the verbose parameter where deprecated.
- *Github Info*: PR `363 <https://github.com/amath-idm/synthpops/pull/363>`__


Version 1.5.2 (2021-03-09)
--------------------------
- *Feature*: Added metadata to pop objects.
- Updated installation instructions and reference citation.
- *Github Info*: PR `365 <https://github.com/amath-idm/synthpops/pull/365>`__, `351 <https://github.com/amath-idm/synthpops/pull/351>`__



