"""Testing the different school mixing types."""
import sciris as sc
import synthpops as sp
import numpy as np
import networkx as nx
import settings
import pytest

# parameters to generate a test population
pars = sc.objdict(
        n                               = settings.pop_sizes.small,
        rand_seed                       = 123,

        with_facilities                 = 1,
        with_non_teaching_staff         = 1,
        with_school_types               = 1,
        average_student_teacher_ratio   = 20,

        school_mixing_type              = 'random',
)


@pytest.mark.parametrize("average_class_size", [12, 25, 70])
def test_random_schools(average_class_size):
    """
    There is a lower bound to the average_class_size and how well clustering
    and density will match when group sizes are small and any change in discrete
    numbers changes these values significantly. It does not mean that the networks
    created are wrong, but rather that at such low sizes, it becomes difficult
    to test their validity with the same thresholds as graphs with higher average
    degrees.
    """
    sp.logger.info("Test random school networks.")
    test_pars = sc.dcp(pars)
    test_pars['average_class_size'] = average_class_size
    pop = sp.Pop(**test_pars)

    G = nx.Graph()

    for i, person in pop.popdict.items():
        if person['scid'] is not None:

            # only for students and teachers
            if (person['sc_student'] == 1) | (person['sc_teacher'] == 1):
                contacts = person['contacts']['S']

                # we only expect this to pass for edges among students and teachers, non teaching staff are added after
                contacts = [c for c in contacts if (pop.popdict[c]['sc_student'] == 1) | (pop.popdict[c]['sc_teacher'] == 1)]
                edges = [(i, c) for c in contacts]
                G.add_edges_from(edges)

    g = [G.subgraph(c) for c in nx.connected_components(G)]  # split into each disconnected school

    for c in range(len(g)):

        expected_density = sp.get_expected_density(average_class_size, len(g[c].nodes()))
        # for Erdos-Renyi random graphs, expected clustering is approximated by the average degree / number of nodes. Expected clustering is the average of the clustering coefficients over all nodes in the graph, where node i has clustering coefficient: 2 * ei / ki * (ki - 1).
        expected_clustering = average_class_size / len(g[c].nodes())
        expected_clustering = min(expected_clustering, 1)
        density = nx.density(g[c])
        clustering = nx.transitivity(g[c])

        lowerbound = 0.85
        upperbound = 1.15
        # check overall density of edges is as expected
        assert expected_density * lowerbound < density < expected_density * upperbound, f'Check failed on random graph densities. {len(g[c].nodes())} {len(g[c].edges())}'
        # check that the distribution of edges is random and clustered as expected for a random graph
        assert expected_clustering * lowerbound < clustering < expected_clustering * upperbound, f'Check failed on random graph clustering. {clustering} {density} {expected_density}, {expected_clustering} {np.mean([g[c].degree(n) for n in g[c].nodes()])}'
        print(f"Check passed. School {c}, size {len(g[c].nodes())} with random mixing has clustering {clustering:.3f} and density {density:.3f} close to expected values {expected_density:.3f}, {expected_clustering:.3f}.")


if __name__ == '__main__':
    pytest.main(['-vs', __file__])
