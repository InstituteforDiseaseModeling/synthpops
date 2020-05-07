=============
|SP| overview
=============

Fundamentally, the population network can be considered a multilayer network_ with
the following qualities:

.. _network: https://en.wikipedia.org/wiki/Network_theory

-   Nodes are people, with attributes like age.
-   Edges represent interactions between people, with attributes like the setting in which the
    interactions take place (for example, household, school, or work). The relationship between the
    interaction setting and properties governing disease transmission, such as frequency of contact and risk
    associated with each contact, is mapped separately by |Cov_s| or other :term:`agent-based model`.
    |SP| reports whether the edge exists or not.

If you are using |SP| with |Cov_s|, note that the relevant value in |Cov_s| is the parameter **beta**,
which captures the probability of transmission via a given edge per :term:`time step`. The value of this parameter
captures both number of effective contacts for disease transmission and transmission probability per contact.

The generated network is a multilayer network in the sense that it is possible for people to be
connected by multiple edges each in different layers of the network. The layers are referred to as
:term:`contact layers`. For example, the :term:`workplace contact layer` is a representation of all
of the pairwise connections between people at work, and the :term:`household contact layer`
represents the pairwise connections between household members. Typically these networks are
clustered; in other words, everyone within a household interacts with each other, but not with other
households. However, they may interact with members of other households via their school or
workplace. Some level of community contacts outside of these networks can be configured using |Cov_s|
or other model being used with |SP|.

|SP| functions in two stages:

#.  Generate people living in households, and then assign individuals to workplaces and schools.
    Save the output to a cache file on disk. Implemented in
    :py:func:`~synthpops.contact_networks.generate_synthetic_population`.
#.  Load the cached file and produce a dictionary that can be used by |Cov_s|. Implemented in
    :py:func:`~synthpops.api.make_population`. |Cov_s| assigns community contacts at random on a daily basis
    to reflect the random and stochastic aspect of contacts in many public spaces, such as shopping
    centers, parks, and community centers.
