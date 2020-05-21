==========
Using |SP|
==========

The overall |SP| workflow is contained in
:py:func:`~synthpops.contact_networks.generate_synthetic_population` and is described below.
The population is generated through households, not a pool of people.

You can provide required data to |SP| in a variety of formats including .csv, .txt, or Microsoft Excel
(.xlsx).


#.  Instantiate a collection of households with sizes drawn from census data. Populations
    cannot be created outside of the :term:`household contact layer`.

#.  For each household, sample the age of a "reference" person from data that maps household
    size to a reference person in those households. The reference person may be referred to as the head
    of the household, a parent in the household, or some other definition specific to the data being used.
    If no data mapping household size to ages of reference individuals are available, then the age of the
    reference person is sampled from the age distribution of adults for the location.

#.  The age bin of the reference people identifies the row of the contact matrix for that location.
    The remaining household members are then selected by sampling an age for the distribution of contacts
    for the reference person's age (in other words, normalizing the values of the row and sampling for a column)
    and assigning someone with that age to the household.

#.  As households are generated, individuals are given IDs.

#.  After households are constructed, students are chosen according to enrollment data by age to
    generate the :term:`school contact layer`.

#.  Students are assigned to schools using a similar method as above, where we select the age of a
    reference person and then select their contacts in school from an age-specific contact matrix for
    the school setting and data on school sizes.

#.  With all students assigned to schools, teachers are selected from the labor force according to
    employment data.

#.  The rest of the labor force are assigned to workplaces in the :term:`workplace contact layer` by
    selecting a reference person and their contacts using an age-specific contact matrix and data on
    workplace sizes.

Examples
========

Examples live in the *examples* folder. These can be run as follows:

*   ``python examples/make_generic_contacts.py``

    Creates a dictionary of individuals, each of whom are represented by another dictionary with
    their contacts contained in the ``contacts`` key. Contacts are selected at random with degree
    distribution following the Erdos-Renyi graph model.

*   ``python examples/generate_contact_network_with_microstructure.py``

    Creates and saves to file households, schools, and workplaces of individuals with unique IDs,
    and a table mapping IDs to ages. Two versions of each contact layer (households, schools, or
    workplaces) are saved; one with the unique IDs of each individual in each group (a single
    household, school or workplace), and one with their ages (for easy viewing of the age mixing
    patterns created).

*   ``python examples/load_contacts_and_show_some_layers.py``

    Loads a multilayer contact network made of three layers and shows the age and ages of
    contacts for the first 20 people.

In the *tests* folder, you can view the following to see examples of additional functionality.

*  ``test_synthpop.py``

    Reads in demographic data and generates populations matching those demographics.

*   ``test_contacts.py``

    Generates random contact networks with individuals matching demographic data or reads in
    synthetic contact networks with three layers (households, schools, and workplaces).

*   ``test_contact_network_generation.py``

    Generates synthetic contact networks  in households, schools, and workplaces with Seattle
    Metro data (and writes to file).

The other topics in this section walk through the specific data sources and details about the settings
for each of the :term:`contact layers`.

.. toctree::

    households
    schools
    workplaces