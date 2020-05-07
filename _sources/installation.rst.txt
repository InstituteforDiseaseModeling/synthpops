============
Installation
============

Follow the instructions below to install |SP|.

Requirements
============

|Python_supp|. (Note: Python 2 is not supported.)

We also recommend, but do not require, using Python virtual environments. For
more information, see documentation for venv_ or Anaconda_.

.. _venv: https://docs.python.org/3/tutorial/venv.html
.. _Anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Installation
============

Complete the following steps to install |SP|:

#.  Fork and clone the |SP| `GitHub repository`_.
#.  Open a command prompt and navigate to the |SP| directory.
#.  Run the following script::

        python setup.py develop

Load data
=========

.. note::

    This module needs to load in data in order to function.


To set the data location, add the following to your scripts::

    import synthpops as sp
    sp.set_datadir('my-data-folder')


The data folder will need to have files in this kind of structure::

    demographics/
    contact_matrices_152_countries/

You can find provided data in this format under the *data* folder in the |SP| `GitHub repository`_.

.. _GitHub repository: https://github.com/InstituteforDiseaseModeling/synthpops

Quick start guide
=================

The following code creates a synthetic population for Seattle, Washington::

    import synthpops as sp

    sp.validate()

    datadir = sp.datadir # this should be where your demographics data folder resides

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'
    level = 'county'

    npop = 10000 # how many people in your population
    sp.generate_synthetic_population(npop,datadir,location=location,
                                     state_location=state_location,country_location=country_location,
                                     sheet_name=sheet_name,level=level)

