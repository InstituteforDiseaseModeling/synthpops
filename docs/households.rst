=======================
Household contact layer
=======================

The :term:`household contact layer` represents the pairwise connections between household members.
The population is generated within this contact layer, not as a separate pool of people.

As locations, households are special in the following ways:

-   Unlike schools and workplaces, everyone must be assigned to a household.
-   The size of the household is important (for example, a 2-person household looks very different in
    comparison to a 5- or 6-person household) and some households only have 1 person.
-   The reference person/head of the household can be well-defined by data.


Data needed
===========

The following data sets are required for households:

#.  **Age bracket distribution** specifying the distribution of people in age bins for the location.
    For example::

        age_bracket , percent
        0_4         , 0.0594714358950416
        5_9         , 0.06031137308234759
        10_14       , 0.05338015778985113
        15_19       , 0.054500690394160285
        20_24       , 0.06161403846144956
        25_29       , 0.08899312471888453
        30_34       , 0.0883533486774803
        35_39       , 0.07780767611060545
        40_44       , 0.07099017823587304
        45_49       , 0.06996903280562596
        50_54       , 0.06655242534751997
        55_59       , 0.06350008343899961
        60_64       , 0.05761405140489549
        65_69       , 0.04487122889235999
        70_74       , 0.030964420778483555
        75_100       , 0.05110673396642193

#.  **Age distribution of the reference person for each household size**

    The distribution is what matters, so it doesn't matter if absolute counts are available or not,
    each *row* is normalized. If this is not available, default to sampling the age of the reference
    individual from the age distribution for adults::

        family_size , 18-20 , 20-24 , 25-29 , 30-34 , 35-39 , 40-44 , 45-49 , 50-54 , 55-64 , 65-74 , 75-99
        2           , 163   , 999   , 2316  , 2230  , 1880  , 1856  , 2390  , 3118  , 9528  , 9345  , 5584
        3           , 115   , 757   , 1545  , 1907  , 2066  , 1811  , 2028  , 2175  , 3311  , 1587  , 588
        4           , 135   , 442   , 1029  , 1951  , 2670  , 2547  , 2368  , 1695  , 1763  , 520   , 221
        5           , 61    , 172   , 394   , 905   , 1429  , 1232  , 969   , 683   , 623   , 235   , 94
        6           , 25    , 81    , 153   , 352   , 511   , 459   , 372   , 280   , 280   , 113   , 49
        7           , 24    , 33    , 63    , 144   , 279   , 242   , 219   , 115   , 157   , 80    , 16

#.  **Distribution of household sizes**::


        household_size , percent
        1              , 0.2781590909877753
        2              , 0.3443313103056699
        3              , 0.15759535523004006
        4              , 0.13654311541644018
        5              , 0.050887858718118274
        6              , 0.019738368167953997
        7              , 0.012744901174002305

#.  **Household contact matrix** specifying the number/weight of contacts by age bin::

                0-10        , 10-20       , 20-30
        0-10    0.659867911 , 0.503965302 , 0.214772978
        10-20   0.314776879 , 0.895460015 , 0.412465791
        20-30   0.132821425 , 0.405073038 , 1.433888594

    By default, |SP| uses matrices from a study (`Prem et al. 2017`_) that projected inferred age mixing
    patterns from the POLYMOD study (`Mossong et al. 2008`_) in Europe to other countries. |SP|
    can take in user-specified contact matrices if other age mixing patterns are available for the
    household, school, and workplace settings (see the social contact data on Zenodo_ for other
    empirical contact matrices from survey studies).

    In theory, the household contact matrix varies with household size, but in general data at that resolution is unavailable.

Workflow
========

Use these |SP| functions to instantiate households as follows:

#.  Call :py:func:`~synthpops.contact_networks.generate_synthetic_population` and provide the binned
    age bracket distribution data described above. This wrapper function calls the following functions:

    #.  From the binned age distribution, :py:func:`~synthpops.synthpops.get_age_n` creates samples
        of ages from the binned distribution, and then normalizes to create a single-year distribution.
        This distribution can therefore be gathered using whatever age bins are present in any given dataset.

    #.  :py:func:`~synthpops.contact_networks.generate_household_sizes_from_fixed_pop_size` generates empty
        households with known size based on the distribution of household sizes.

    #.  :py:func:`~synthpops.contact_networks.generate_all_households` contains the core implementation
        and constructs households with individuals of different ages living together. It takes in the
        remaining data sources above, and then does the following:

        -   Calls :py:func:`~synthpops.contact_networks.generate_living_alone` to populate households with
            1 person (either from data on those living alone or, if unavailable, from the adult age distribution).
        -   Calls :py:func:`~synthpops.contact_networks.generate_larger_households` repeatedly with with
            different household sizes to populate those households, first sampling the age of a reference
            person and then their household contacts as outlined above.

.. _Mossong et al. 2008: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074
.. _Fumanelli et al. 2012: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002673
.. _Prem et al. 2017: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697
.. _Mistry et al. 2020: https://arxiv.org/abs/2003.01214
.. _Zenodo: https://zenodo.org/communities/social_contact_data/?page=1&size=20