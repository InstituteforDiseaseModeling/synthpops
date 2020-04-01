import sciris as sc
import numpy as np
import pandas as pd
from numba import jit
import random
import itertools
import synthpops as sp

contact_size = {'S': 20, 'W': 10, 'H': 0}

def rehydrate(data):
    popdict = sc.dcp(data['popdict'])
    mapping = {'H':'households', 'S':'schools', 'W':'workplaces'}
    for key,label in mapping.items():
        for r in data[label]: # House, school etc
            for uid in r:
                current_contacts = len(popdict[uid]['contacts'][key])
                if contact_size[key]:  # 0 for unlimited
                    to_select = contact_size[key] - current_contacts
                    if to_select <= 0:  # already filled list from other actors
                        continue
                    contacts = np.random.choice(r, size=to_select)
                else:
                    contacts = r
                for c in contacts:
                    if c == uid:
                        continue
                    if c in popdict[uid]['contacts'][key]:
                        continue
                    popdict[uid]['contacts'][key].add(c)
                    popdict[c]['contacts'][key].add(uid)
    return popdict

connection_proba = {
    'households': 1.,
    'schools': 0.1,
    'workplaces': 0.1
}

relative_infection_proba = {
    'households': 1.,
    'schools': 0.3,
    'workplaces': 0.4
}


# Generate cluster with randomization, schools, workplaces etc
@jit(nopython=True, parallel=True)
def generate_random_connect_cluster(nodes, connection_proba, relative_infection_proba, cluster_type):
    # Generate connections in cluster with given probability
    all_connections = np.random.random((nodes.shape[0], nodes.shape[0])) < connection_proba
    # Symmetrize matrix, this will ensure that all connections are bi-directional
    all_connections = np.maximum(all_connections, all_connections.T)
    connections = []
    for i in range(all_connections.shape[0]):
        for j in range(all_connections.shape[0]):
            if all_connections[i][j]:
                connections.append((nodes[i], nodes[j], relative_infection_proba, cluster_type))
    return connections

# Generate fully connected cluster, households
def generate_full_connect_cluster(nodes, relative_infection_proba, cluster_type):
    permutations = itertools.permutations(nodes, 2)
    return list(map(lambda p: p + (relative_infection_proba, cluster_type), permutations))

def hydrate_edgelist(data):
    edgelist = []
    for hcluster in data["households"]:
        edgelist.extend(generate_full_connect_cluster(np.array(hcluster), relative_infection_proba["households"], "households"))
    for scluster in data["schools"]:
        edgelist.extend(generate_random_connect_cluster(np.array(scluster), connection_proba["schools"], relative_infection_proba["schools"], "schools"))
    for wcluster in data["workplaces"]:
        edgelist.extend(generate_random_connect_cluster(np.array(wcluster), connection_proba["workplaces"], relative_infection_proba["workplaces"], "workplaces"))
    return pd.DataFrame(edgelist, columns=["person1", "person2", "relative_infection_proba", "cluster_type"]).drop_duplicates()

fn = '../data/synthpop_122000.pop'
data = sc.loadobj(fn)

sc.tic()
popdict = rehydrate(data)
# popdict = sp.trim_contacts(popdict)
sc.toc()
sc.tic()
edgelist = hydrate_edgelist(data=data)
sc.toc()