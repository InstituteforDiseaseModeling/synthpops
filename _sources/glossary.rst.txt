========
Glossary
========

.. glossary::

    contact layers
        Each of the layers of the population network that is a representation of all of the
        pairwise connections between people in a given location, such as school, work, or households.

    node
        In `network theory`_, the discrete object being represented. In |SP|, nodes represent
        people and can have attributes like age assigned.

    edge
        In `network theory`_, the interactions between discrete objects. In |SP|, edges represent
        interactions between people, with attributes like the setting in which the interactions take
        place (for example, household, school, or work). The relationship between the
        interaction setting and properties governing disease transmission, such as frequency of
        contact and risk associated with each contact, is mapped separately by |Cov_s| or other
        :term:`agent-based model`. |SP| reports whether the edge exists or not.

    agent-based model
        A type of simulation that models the actions and interactions of autonomous agents (both
        individual and collective entities such as organizations or groups).

    time step
        A discrete number of hours or days in which the “simulation states” of all “simulation
        objects” (interventions, infections, immune systems, or individuals) are updated in a
        simulation. Each time step will complete processing before launching the next one. For
        example, a time step would process the migration data for populations moving between nodes
        via rail, airline, and road. The migration of individuals between nodes is the last step of
        the time step after updating states.

    household contact layer
        The layer in the population network that represents all of the pairwise connections between
        people in households. All people must be part of the household contact layer, though some
        households may consist of a single person.

    school contact layer
        The layer in the population network that represents all of the pairwise connections between
        people in schools. This includes both students and teachers. The school and workplace contact
        layers are mutually exclusive, someone cannot be both a student and a worker.

    workplace contact layer
        The layer in the population network that represents all of the pairwise connections between
        people in workplaces excluding teachers in schools. The school and workplace contact
        layers are mutually exclusive, someone cannot be both a student and a worker.

.. _network theory: https://en.wikipedia.org/wiki/Network_theory