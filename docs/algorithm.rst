==============
|SP| algorithm
==============

This topic describes the algorithm used by |SP| to generate the connections between people in each
of the  :term:`contact layers` for a given location in the real world. The fundamental algorithm is
the same for homes, schools, and workplaces, but with some variations for each.

The method draws upon the following previously published models to infer
high-resolution age-specific contact patterns in different physical settings and
locations:

* `Mossong et al. 2008`_
* `Fumanelli et al. 2012`_
* `Prem et al. 2017`_
* `Mistry et al. 2020`_

The general idea is to use age-specific contact matrices that describe age mixing patterns for a specific 
population. By default, |SP| uses Prem et al.’s (2017) matrices, which project inferred age mixing patterns
from the POLYMOD study (Mossong et al. 2008) in Europe to other countries. However, user-specified contact 
matrices can also be implemented for customizing age mixing patterns for the household, school, and workplace 
settings (see the social contact data on `Zenodo <https://zenodo.org/communities/social_contact_data>`_ for other empirical contact matrices 
from survey studies).

The matrices represent the average number of contacts between people for different age bins (the
default matrices use 5-year age bins). For example, a household of two individuals is relatively
unlikely to consist of a 25-year-old and a 15-year-old, so for the 25-29 year age bin in the
household layer, there are a low number of expected contacts with the 15-19 year age bin (c.f., Fig.
2c in Prem et al.).

.. _Mossong et al. 2008: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074
.. _Fumanelli et al. 2012: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002673
.. _Prem et al. 2017: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697
.. _Mistry et al. 2020: https://arxiv.org/abs/2003.01214


