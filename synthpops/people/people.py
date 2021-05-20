'''
Alternate representation of a population as a People object. Originally based on
the corresponding Covasim classes and functions.
'''

import numpy as np
import pylab as pl
import sciris as sc
import pandas as pd
import networkx as nx
from . import utils as spu
from .. import version as spv

__all__ = ['FlexPretty', 'BasePeople', 'Person', 'FlexDict', 'Contacts', 'Layer', 'People']


#%% Define people classes

class FlexPretty(sc.prettyobj):
    '''
    A class that supports multiple different display options: namely obj.brief()
    for a one-line description and obj.disp() for a full description.

    New in version 1.10.0.
    '''

    def __repr__(self):
        ''' Use brief repr by default '''
        try:
            string = self._brief()
        except Exception as E:
            string = sc.objectid(self)
            string += f'Warning, something went wrong printing object:\n{str(E)}'
        return string

    def _disp(self):
        ''' Verbose output -- use Sciris' pretty repr by default '''
        return sc.prepr(self)

    def disp(self, output=False):
        ''' Print or output verbose representation of the object '''
        string = self._disp()
        if not output:
            print(string)
        else:
            return string

    def _brief(self):
        ''' Brief output -- use a one-line output, a la Python's default '''
        return sc.objectid(self)

    def brief(self, output=False):
        ''' Print or output a brief representation of the object '''
        string = self._brief()
        if not output:
            print(string)
        else:
            return string


class BasePeople(FlexPretty):
    '''
    A class to handle all the boilerplate for people -- note that as with the
    BaseSim vs Sim classes, everything interesting happens in the People class,
    whereas this class exists to handle the less interesting implementation details.

    New in version 1.10.0.
    '''

    def __init__(self):
        super().__init__()
        self._lock = False # Prevent further modification of keys
        self._keys = []
        return


    def __getitem__(self, key):
        ''' Allow people['attr'] instead of getattr(people, 'attr')
            If the key is an integer, alias `people.person()` to return a `Person` instance
        '''

        try:
            return self.__dict__[key]
        except: # pragma: no cover
            if isinstance(key, int):
                return self.person(key)
            else:
                errormsg = f'Key "{key}" is not a valid attribute of people'
                raise AttributeError(errormsg)


    def __setitem__(self, key, value):
        ''' Ditto '''
        if self._lock and key not in self.__dict__: # pragma: no cover
            errormsg = f'Key "{key}" is not a current attribute of people, and the people object is locked; see people.unlock()'
            raise AttributeError(errormsg)
        self.__dict__[key] = value
        self._keys.append(key)
        return


    def keys(self):
        ''' Get the keys that have been set '''
        curr_keys = self.__dict__.keys()
        new_keys = [k for k in self._keys if k in curr_keys] # Remove any keys that have been removed
        self._keys = new_keys # Trim any that were removed
        return sc.dcp(new_keys)


    def lock(self):
        ''' Lock the people object to prevent keys from being added '''
        self._lock = True
        return


    def unlock(self):
        ''' Unlock the people object to allow keys to be added '''
        self._lock = False
        return


    def __len__(self):
        ''' This is just a scalar, but validate() and _resize_arrays() make sure it's right '''
        return int(self.pars['pop_size'])


    def __iter__(self):
        ''' Iterate over people '''
        for i in range(len(self)):
            yield self[i]


    def __add__(self, people2):
        ''' Combine two people arrays '''
        newpeople = sc.dcp(self)
        keys = list(self.keys())
        for key in keys:
            npval = newpeople[key]
            p2val = people2[key]
            if npval.ndim == 1:
                newpeople.set(key, np.concatenate([npval, p2val], axis=0), die=False) # Allow size mismatch
            elif npval.ndim == 2:
                newpeople.set(key, np.concatenate([npval, p2val], axis=1), die=False)
            else:
                errormsg = f'Not sure how to combine arrays of {npval.ndim} dimensions for {key}'
                raise NotImplementedError(errormsg)

        # Validate
        newpeople.pars['pop_size'] += people2.pars['pop_size']
        newpeople.validate()

        # Reassign UIDs so they're unique
        newpeople.set('uid', np.arange(len(newpeople)))

        return newpeople


    def __radd__(self, people2):
        ''' Allows sum() to work correctly '''
        if not people2: return self
        else:           return self.__add__(people2)


    def _brief(self):
        '''
        Return a one-line description of the people -- used internally and by repr();
        see people.brief() for the user version.
        '''
        try:
            layerstr = ', '.join([str(k) for k in self.layer_keys()])
            string   = f'People(n={len(self):0n}; layers: {layerstr})'
        except Exception as E: # pragma: no cover
            string = sc.objectid(self)
            string += f'Warning, multisim appears to be malformed:\n{str(E)}'
        return string


    def summarize(self, output=False):
        ''' Print a summary of the people -- same as brief '''
        return self.brief(output=output)


    def set(self, key, value, die=True):
        ''' Ensure sizes and dtypes match '''
        current = self[key]
        value = np.array(value, dtype=current.dtype) # Ensure it's the right type
        if die and len(value) != len(current): # pragma: no cover
            errormsg = f'Length of new array does not match current ({len(value)} vs. {len(current)})'
            raise IndexError(errormsg)
        self[key] = value
        return


    def get(self, key):
        ''' Convenience method -- key can be string or list of strings '''
        if isinstance(key, str):
            return self[key]
        elif isinstance(key, list):
            arr = np.zeros((len(self), len(key)))
            for k,ky in enumerate(key):
                arr[:,k] = self[ky]
            return arr


    def true(self, key):
        ''' Return indices matching the condition '''
        return self[key].nonzero()[0]


    def false(self, key):
        ''' Return indices not matching the condition '''
        return (~self[key]).nonzero()[0]


    def defined(self, key):
        ''' Return indices of people who are not-nan '''
        return (~np.isnan(self[key])).nonzero()[0]


    def undefined(self, key):
        ''' Return indices of people who are nan '''
        return np.isnan(self[key]).nonzero()[0]


    def count(self, key):
        ''' Count the number of people for a given key '''
        return (self[key]>0).sum()


    def count_not(self, key):
        ''' Count the number of people who do not have a property for a given key '''
        return (self[key]==0).sum()


    def set_pars(self, pars=None):
        '''
        Re-link the parameters stored in the people object to the sim containing it,
        and perform some basic validation.
        '''
        if pars is None:
            pars = {}
        elif sc.isnumber(pars): # Interpret as a population size
            pars = {'pop_size':pars} # Ensure it's a dictionary
        orig_pars = self.__dict__.get('pars') # Get the current parameters using dict's get method
        pars = sc.mergedicts(orig_pars, pars)
        if 'pop_size' not in pars:
            errormsg = f'The parameter "pop_size" must be included in a population; keys supplied were:\n{sc.newlinejoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg)
        pars['pop_size'] = int(pars['pop_size'])
        pars.setdefault('location', None)
        self.pars = pars # Actually store the pars
        return


    def layer_keys(self):
        ''' Get the available contact keys -- try contacts first, then beta_layer '''
        try:
            keys = list(self.contacts.keys())
        except: # If not fully initialized
            try:
                keys = list(self.pars['contacts'].keys())
            except:  # pragma: no cover # If not even partially initialized
                keys = []
        return keys


    def indices(self):
        ''' The indices of each people array '''
        return np.arange(len(self))


    def validate(self, die=True, verbose=False):

        # Check that the keys match
        contact_layer_keys = set(self.contacts.keys())
        layer_keys    = set(self.layer_keys())
        if contact_layer_keys != layer_keys:
            errormsg = f'Parameters layers {layer_keys} are not consistent with contact layers {contact_layer_keys}'
            raise ValueError(errormsg)

        # Check that the length of each array is consistent
        expected_len = len(self)
        for key in self.keys():
            actual_len = len(self[key])
            if actual_len != expected_len: # pragma: no cover
                if die:
                    errormsg = f'Length of key "{key}" did not match population size ({actual_len} vs. {expected_len})'
                    raise IndexError(errormsg)
                else:
                    if verbose:
                        print(f'Resizing "{key}" from {actual_len} to {expected_len}')
                    self._resize_arrays(keys=key)

        # Check that the layers are valid
        for layer in self.contacts.values():
            layer.validate()

        return


    def _resize_arrays(self, new_size=None, keys=None):
        ''' Resize arrays if any mismatches are found '''

        # Handle None or tuple input
        if new_size is None:
            new_size = len(self)
        pop_size = new_size if not isinstance(new_size, tuple) else new_size[1]
        self.pars['pop_size'] = pop_size

        # Reset sizes
        if keys is None:
            keys = self.keys()
        keys = sc.promotetolist(keys)
        for key in keys:
            self[key].resize(new_size, refcheck=False) # Don't worry about cross-references to the arrays

        return


    def to_df(self):
        ''' Convert to a Pandas dataframe '''
        df = pd.DataFrame.from_dict({key:self[key] for key in self.keys()})
        return df


    def to_arr(self):
        ''' Return as numpy array '''
        arr = np.empty((len(self), len(self.keys())), dtype=spu.default_float)
        for k,key in enumerate(self.keys()):
            if key == 'uid':
                arr[:,k] = np.arange(len(self))
            else:
                arr[:,k] = self[key]
        return arr


    def person(self, ind):
        ''' Method to create person from the people '''
        p = Person()
        for key in self.keys():
            data = self[key]
            if data.ndim == 1:
                val = data[ind]
            elif data.ndim == 2:
                val = data[:,ind]
            else:
                errormsg = f'Cannot extract data from {key}: unexpected dimensionality ({data.ndim})'
                raise ValueError(errormsg)
            setattr(p, key, val)

        contacts = {}
        for lkey, layer in self.contacts.items():
            contacts[lkey] = layer.find_contacts(ind)
        p.contacts = contacts

        return p


    def to_people(self):
        ''' Return all people as a list '''
        return list(self)


    def from_people(self, people, resize=True):
        ''' Convert a list of people back into a People object '''

        # Handle population size
        pop_size = len(people)
        if resize:
            self._resize_arrays(new_size=pop_size)

        # Iterate over people -- slow!
        for p,person in enumerate(people):
            for key in self.keys():
                self[key][p] = getattr(person, key)

        return


    def to_graph(self, full_output=False): # pragma: no cover
        '''
        Convert all people to a networkx MultiDiGraph, including all properties of
        the people (nodes) and contacts (edges).

        Args:
            full_output (bool): if true, return nodes and edges along with the graph object
        '''

        # Copy data from people into graph
        G = self.contacts.to_graph()
        for key in self.keys():
            data = {k:v for k,v in enumerate(self[key])}
            nx.set_node_attributes(G, data, name=key)

        # Include global layer weights
        for u,v,k in G.edges(keys=True):
            edge = G[u][v][k]
            try:
                edge['beta'] *= self.pars['beta_layer'][edge['layer']]
            except:
                pass

        if not full_output:
            return G
        else:
            nodes = G.nodes(data=True)
            edges = G.edges(keys=True)
            return G, nodes, edges


    def init_contacts(self, reset=False):
        ''' Initialize the contacts dataframe with the correct columns and data types '''

        # Create the contacts dictionary
        contacts = Contacts(layer_keys=self.layer_keys())

        if self.contacts is None or reset: # Reset all
            self.contacts = contacts
        else: # Only replace specified keys
            for key,layer in contacts.items():
                self.contacts[key] = layer
        return


    def add_contacts(self, contacts, lkey=None, beta=None):
        '''
        Add new contacts to the array. See also contacts.add_layer().
        '''

        # If no layer key is supplied and it can't be worked out from defaults, use the first layer
        if lkey is None:
            lkey = self.layer_keys()[0]

        # Validate the supplied contacts
        if isinstance(contacts, Contacts):
            new_contacts = contacts
        elif isinstance(contacts, Layer):
            new_contacts = {}
            new_contacts[lkey] = contacts
        elif sc.checktype(contacts, 'array'):
            new_contacts = {}
            new_contacts[lkey] = pd.DataFrame(data=contacts)
        elif isinstance(contacts, dict):
            new_contacts = {}
            new_contacts[lkey] = pd.DataFrame.from_dict(contacts)
        elif isinstance(contacts, list): # Assume it's a list of contacts by person, not an edgelist
            new_contacts = self.make_edgelist(contacts) # Assume contains key info
        else: # pragma: no cover
            errormsg = f'Cannot understand contacts of type {type(contacts)}; expecting dataframe, array, or dict'
            raise TypeError(errormsg)

        # Ensure the columns are right and add values if supplied
        for lkey, new_layer in new_contacts.items():
            n = len(new_layer['p1'])
            if 'beta' not in new_layer.keys() or len(new_layer['beta']) != n:
                if beta is None:
                    beta = 1.0
                beta = spu.default_float(beta)
                new_layer['beta'] = np.ones(n, dtype=spu.default_float)*beta

            # Create the layer if it doesn't yet exist
            if lkey not in self.contacts:
                self.contacts[lkey] = Layer(label=lkey)

            # Actually include them, and update properties if supplied
            for col in self.contacts[lkey].keys(): # Loop over the supplied columns
                self.contacts[lkey][col] = np.concatenate([self.contacts[lkey][col], new_layer[col]])
            self.contacts[lkey].validate()

        return


    def make_edgelist(self, contacts):
        '''
        Parse a list of people with a list of contacts per person and turn it
        into an edge list.
        '''

        # Handle layer keys
        lkeys = self.layer_keys()
        if len(contacts):
            contact_keys = contacts[0].keys() # Pull out the keys of this contact list
            lkeys += [key for key in contact_keys if key not in lkeys] # Extend the layer keys

        # Initialize the new contacts
        new_contacts = Contacts(layer_keys=lkeys)
        for lkey in lkeys:
            new_contacts[lkey]['p1']    = [] # Person 1 of the contact pair
            new_contacts[lkey]['p2']    = [] # Person 2 of the contact pair

        # Populate the new contacts
        for p,cdict in enumerate(contacts):
            for lkey,p_contacts in cdict.items():
                n = len(p_contacts) # Number of contacts
                new_contacts[lkey]['p1'].extend([p]*n) # e.g. [4, 4, 4, 4]
                new_contacts[lkey]['p2'].extend(p_contacts) # e.g. [243, 4538, 7,19]

        # Turn into a dataframe
        for lkey in lkeys:
            new_layer = Layer(label=lkey)
            for ckey,value in new_contacts[lkey].items():
                new_layer[ckey] = np.array(value, dtype=new_layer.meta[ckey])
            new_contacts[lkey] = new_layer

        return new_contacts


    @staticmethod
    def remove_duplicates(df):
        ''' Sort the dataframe and remove duplicates -- note, not extensively tested '''
        p1 = df[['p1', 'p2']].values.min(1) # Reassign p1 to be the lower-valued of the two contacts
        p2 = df[['p1', 'p2']].values.max(1) # Reassign p2 to be the higher-valued of the two contacts
        df['p1'] = p1
        df['p2'] = p2
        df.sort_values(['p1', 'p2'], inplace=True) # Sort by p1, then by p2
        df.drop_duplicates(['p1', 'p2'], inplace=True) # Remove duplicates
        df = df[df['p1'] != df['p2']] # Remove self connections
        df.reset_index(inplace=True, drop=True)
        return df


class Person(sc.prettyobj):
    '''
    Class for a single person. Note: this is largely deprecated since People
    is now based on arrays rather than being a list of people.

    New in version 1.10.0.
    '''
    def __init__(self, pars=None, uid=None, age=-1, sex=-1, contacts=None):
        self.uid         = uid # This person's unique identifier
        self.age         = spu.default_float(age) # Age of the person (in years)
        self.sex         = spu.default_int(sex) # Female (0) or male (1)
        self.contacts    = contacts # Contacts
        return


class FlexDict(dict):
    '''
    A dict that allows more flexible element access: in addition to obj['a'],
    also allow obj[0]. Lightweight implementation of the Sciris odict class.
    '''

    def __getitem__(self, key):
        ''' Lightweight odict -- allow indexing by number, with low performance '''
        try:
            return super().__getitem__(key)
        except KeyError as KE:
            try: # Assume it's an integer
                dictkey = self.keys()[key]
                return self[dictkey]
            except:
                raise sc.KeyNotFoundError(KE) # Raise the original error

    def keys(self):
        return list(super().keys())

    def values(self):
        return list(super().values())

    def items(self):
        return list(super().items())


class Contacts(FlexDict):
    '''
    A simple (for now) class for storing different contact layers.

    New in version 1.10.0.
    '''
    def __init__(self, layer_keys=None):
        if layer_keys is not None:
            for lkey in layer_keys:
                self[lkey] = Layer(label=lkey)
        return

    def __repr__(self):
        ''' Use slightly customized repr'''
        keys_str = ', '.join([str(k) for k in self.keys()])
        output = f'Contacts({keys_str})\n'
        for key in self.keys():
            output += f'\n"{key}": '
            output += self[key].__repr__() + '\n'
        return output


    def __len__(self):
        ''' The length of the contacts is the length of all the layers '''
        output = 0
        for key in self.keys():
            try:
                output += len(self[key])
            except: # pragma: no cover
                pass
        return output


    def add_layer(self, **kwargs):
        '''
        Small method to add one or more layers to the contacts. Layers should
        be provided as keyword arguments.

        **Example**::

            hospitals_layer = cv.Layer(label='hosp')
            sim.people.contacts.add_layer(hospitals=hospitals_layer)
        '''
        for lkey,layer in kwargs.items():
            layer.validate()
            self[lkey] = layer
        return


    def pop_layer(self, *args):
        '''
        Remove the layer(s) from the contacts.

        **Example**::

            sim.people.contacts.pop_layer('hospitals')

        Note: while included here for convenience, this operation is equivalent
        to simply popping the key from the contacts dictionary.
        '''
        for lkey in args:
            self.pop(lkey)
        return


    def to_graph(self): # pragma: no cover
        '''
        Convert all layers to a networkx MultiDiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=50, pop_type='hybrid').run()
            G = sim.people.contacts.to_graph()
            nx.draw(G)
        '''
        import networkx as nx
        H = nx.MultiDiGraph()
        for lkey,layer in self.items():
            G = layer.to_graph()
            H = nx.compose(H, nx.MultiDiGraph(G))
        return H



class Layer(FlexDict):
    '''
    A small class holding a single layer of contact edges (connections) between people.

    The input is typically three arrays: person 1 of the connection, person 2 of
    the connection, and the weight of the connection. Connections are undirected;
    each person is both a source and sink.

    This class is usually not invoked directly by the user, but instead is called
    as part of the population creation.

    Args:
        p1 (array): an array of N connections, representing people on one side of the connection
        p2 (array): an array of people on the other side of the connection
        beta (array): an array of weights for each connection
        label (str): the name of the layer (optional)
        kwargs (dict): other keys copied directly into the layer

    Note that all arguments (except for label) must be arrays of the same length,
    although not all have to be supplied at the time of creation (they must all
    be the same at the time of initialization, though, or else validation will fail).

    **Examples**::

        # Generate an average of 10 contacts for 1000 people
        n = 10_000
        n_people = 1000
        p1 = np.random.randint(n_people, size=n)
        p2 = np.random.randint(n_people, size=n)
        beta = np.ones(n)
        layer = cv.Layer(p1=p1, p2=p2, beta=beta, label='rand')

        # Convert one layer to another with extra columns
        index = np.arange(n)
        self_conn = p1 == p2
        layer2 = cv.Layer(**layer, index=index, self_conn=self_conn, label=layer.label)

    New in version 1.10.0.
    '''

    def __init__(self, label=None, **kwargs):
        self.meta = {
            'p1':    spu.default_int,   # Person 1
            'p2':    spu.default_int,   # Person 2
            'beta':  spu.default_float, # Default transmissibility for this contact type
        }
        self.basekey = 'p1' # Assign a base key for calculating lengths and performing other operations
        self.label = label

        # Initialize the keys of the layers
        for key,dtype in self.meta.items():
            self[key] = np.empty((0,), dtype=dtype)

        # Set data, if provided
        for key,value in kwargs.items():
            self[key] = np.array(value, dtype=self.meta.get(key))

        return


    def __len__(self):
        try:
            return len(self[self.basekey])
        except: # pragma: no cover
            return 0


    def __repr__(self):
        ''' Convert to a dataframe for printing '''
        namestr = self.__class__.__name__
        labelstr = f'"{self.label}"' if self.label else '<no label>'
        keys_str = ', '.join(self.keys())
        output = f'{namestr}({labelstr}, {keys_str})\n' # e.g. Layer("h", p1, p2, beta)
        output += self.to_df().__repr__()
        return output


    def __contains__(self, item):
        """
        Check if a person is present in a layer

        Args:
            item: Person index

        Returns: True if person index appears in any interactions

        """
        return (item in self['p1']) or (item in self['p2'])

    @property
    def members(self):
        """
        Return sorted array of all members
        """
        return np.unique([self['p1'], self['p2']])


    def meta_keys(self):
        ''' Return the keys for the layer's meta information -- i.e., p1, p2, beta '''
        return self.meta.keys()


    def validate(self):
        ''' Check the integrity of the layer: right types, right lengths '''
        n = len(self[self.basekey])
        for key,dtype in self.meta.items():
            if dtype:
                actual = self[key].dtype
                expected = dtype
                if actual != expected:
                    errormsg = f'Expecting dtype "{expected}" for layer key "{key}"; got "{actual}"'
                    raise TypeError(errormsg)
            actual_n = len(self[key])
            if n != actual_n:
                errormsg = f'Expecting length {n} for layer key "{key}"; got {actual_n}'
                raise TypeError(errormsg)
        return


    def pop_inds(self, inds):
        '''
        "Pop" the specified indices from the edgelist and return them as a dict.
        Returns in the right format to be used with layer.append().

        Args:
            inds (int, array, slice): the indices to be removed
        '''
        output = {}
        for key in self.meta_keys():
            output[key] = self[key][inds] # Copy to the output object
            self[key] = np.delete(self[key], inds) # Remove from the original
        return output


    def append(self, contacts):
        '''
        Append contacts to the current layer.

        Args:
            contacts (dict): a dictionary of arrays with keys p1,p2,beta, as returned from layer.pop_inds()
        '''
        for key in self.keys():
            new_arr = contacts[key]
            n_curr = len(self[key]) # Current number of contacts
            n_new = len(new_arr) # New contacts to add
            n_total = n_curr + n_new # New size
            self[key] = np.resize(self[key], n_total) # Resize to make room, preserving dtype
            self[key][n_curr:] = new_arr # Copy contacts into the layer
        return


    def to_df(self):
        ''' Convert to dataframe '''
        df = pd.DataFrame.from_dict(self)
        return df


    def from_df(self, df, keys=None):
        ''' Convert from a dataframe '''
        if keys is None:
            keys = self.meta_keys()
        for key in keys:
            self[key] = df[key].to_numpy()
        return self


    def to_graph(self): # pragma: no cover
        '''
        Convert to a networkx DiGraph

        **Example**::

            import networkx as nx
            sim = cv.Sim(pop_size=20, pop_type='hybrid').run()
            G = sim.people.contacts['h'].to_graph()
            nx.draw(G)
        '''
        import networkx as nx
        data = [np.array(self[k], dtype=dtype).tolist() for k,dtype in [('p1', int), ('p2', int), ('beta', float)]]
        G = nx.DiGraph()
        G.add_weighted_edges_from(zip(*data), weight='beta')
        nx.set_edge_attributes(G, self.label, name='layer')
        return G


    def find_contacts(self, inds, as_array=True):
        """
        Find all contacts of the specified people

        For some purposes (e.g. contact tracing) it's necessary to find all of the contacts
        associated with a subset of the people in this layer. Since contacts are bidirectional
        it's necessary to check both P1 and P2 for the target indices. The return type is a Set
        so that there is no duplication of indices (otherwise if the Layer has explicit
        symmetric interactions, they could appear multiple times). This is also for performance so
        that the calling code doesn't need to perform its own unique() operation. Note that
        this cannot be used for cases where multiple connections count differently than a single
        infection, e.g. exposure risk.

        Args:
            inds (array): indices of people whose contacts to return
            as_array (bool): if true, return as sorted array (otherwise, return as unsorted set)

        Returns:
            contact_inds (array): a set of indices for pairing partners

        Example: If there were a layer with
        - P1 = [1,2,3,4]
        - P2 = [2,3,1,4]
        Then find_contacts([1,3]) would return {1,2,3}
        """

        # Check types
        if not isinstance(inds, np.ndarray):
            inds = sc.promotetoarray(inds)
        if inds.dtype != np.int64:  # pragma: no cover # This is int64 since indices often come from cv.true(), which returns int64
            inds = np.array(inds, dtype=np.int64)

        # Find the contacts
        contact_inds = spu.find_contacts(self['p1'], self['p2'], inds)
        if as_array:
            contact_inds = np.fromiter(contact_inds, dtype=spu.default_int)
            contact_inds.sort()  # Sorting ensures that the results are reproducible for a given seed as well as being identical to previous versions of Covasim

        return contact_inds


    def update(self, people, frac=1.0):
        '''
        Regenerate contacts on each timestep.

        This method gets called if the layer appears in ``sim.pars['dynam_lkeys']``.
        The Layer implements the update procedure so that derived classes can customize
        the update e.g. implementing over-dispersion/other distributions, random
        clusters, etc.

        Typically, this method also takes in the ``people`` object so that the
        update can depend on person attributes that may change over time (e.g.
        changing contacts for people that are severe/critical).

        Args:
            frac (float): the fraction of contacts to update on each timestep
        '''
        # Choose how many contacts to make
        pop_size   = len(people) # Total number of people
        n_contacts = len(self) # Total number of contacts
        n_new = int(np.round(n_contacts*frac)) # Since these get looped over in both directions later
        inds = spu.choose(n_contacts, n_new)

        # Create the contacts, not skipping self-connections
        self['p1'][inds]   = np.array(spu.choose_r(max_n=pop_size, n=n_new), dtype=spu.default_int) # Choose with replacement
        self['p2'][inds]   = np.array(spu.choose_r(max_n=pop_size, n=n_new), dtype=spu.default_int)
        self['beta'][inds] = np.ones(n_new, dtype=spu.default_float)
        return



'''
Defines the Person class and functions associated with making people.
'''


class People(BasePeople):
    '''
    A class to perform all the operations on the people. This class is usually
    not invoked directly, but instead is created automatically by the sim. The
    only required input argument is the population size, but typically the full
    parameters dictionary will get passed instead since it will be needed before
    the People object is initialized.

    Note that this class handles the mechanics of updating the actual people, while
    BasePeople takes care of housekeeping (saving, loading, exporting, etc.). Please
    see the BasePeople class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as pop_size
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    ::Examples::

        ppl1 = cv.People(2000)

        sim = cv.Sim()
        ppl2 = cv.People(sim.pars)

    New in version 1.10.0.
    '''

    def __init__(self, pars, strict=False, **kwargs):
        super().__init__()

        # Handle pars and population size
        self.set_pars(pars)
        self.version = spv.__version__ # Store version info

        # Other initialization
        self.contacts = None
        self.init_contacts() # Initialize the contacts

        # Handle contacts, if supplied (note: they usually are)
        if 'contacts' in kwargs:
            self.add_contacts(kwargs.pop('contacts'))

        # Handle all other values, e.g. age
        for key,value in kwargs.items():
            if strict:
                self.set(key, value)
            else:
                self[key] = value

        return

    #%% Analysis methods

    def plot(self, bins=None, width=1.0, alpha=0.6, fig_args=None, axis_args=None,
                plot_args=None, do_show=None, fig=None):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
            width     (float) : bar width
            font_size (float) : size of font
            alpha     (float) : transparency of the plots
            fig_args  (dict)  : passed to pl.figure()
            axis_args (dict)  : passed to pl.subplots_adjust()
            plot_args (dict)  : passed to pl.plot()
            do_show   (bool)  : whether to show the plot
            fig       (fig)   : handle of existing figure to plot into
        '''
        # Handle inputs
        if bins is None:
            bins = np.arange(0,101)

        # Set defaults
        color     = [0.1,0.1,0.1] # Color for the age distribution
        n_rows    = 4 # Number of rows of plots
        offset    = 0.5 # For ensuring the full bars show up
        gridspace = 10 # Spacing of gridlines
        zorder    = 10 # So plots appear on top of gridlines

        # Handle other arguments
        fig_args  = sc.mergedicts(dict(figsize=(18,11)), fig_args)
        axis_args = sc.mergedicts(dict(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.35), axis_args)
        plot_args = sc.mergedicts(dict(lw=1.5, alpha=0.6, c=color, zorder=10), plot_args)

        # Compute statistics
        min_age = min(bins)
        max_age = max(bins)
        edges = np.append(bins, np.inf) # Add an extra bin to end to turn them into edges
        age_counts = np.histogram(self.age, edges)[0]

        # Create the figure
        if fig is None:
            fig = pl.figure(**fig_args)
        pl.subplots_adjust(**axis_args)

        # Plot age histogram
        pl.subplot(n_rows,2,1)
        pl.bar(bins, age_counts, color=color, alpha=alpha, width=width, zorder=zorder)
        pl.xlim([min_age-offset,max_age+offset])
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.grid(True)
        pl.xlabel('Age')
        pl.ylabel('Number of people')
        pl.title(f'Age distribution ({len(self):n} people total)')

        # Plot cumulative distribution
        pl.subplot(n_rows,2,2)
        age_sorted = sorted(self.age)
        y = np.linspace(0, 100, len(age_sorted)) # Percentage, not hard-coded!
        pl.plot(age_sorted, y, '-', **plot_args)
        pl.xlim([0,max_age])
        pl.ylim([0,100]) # Percentage
        pl.xticks(np.arange(0, max_age+1, gridspace))
        pl.yticks(np.arange(0, 101, gridspace)) # Percentage
        pl.grid(True)
        pl.xlabel('Age')
        pl.ylabel('Cumulative proportion (%)')
        pl.title(f'Cumulative age distribution (mean age: {self.age.mean():0.2f} years)')

        # Calculate contacts
        lkeys = self.layer_keys()
        n_layers = len(lkeys)
        contact_counts = sc.objdict()
        for lk in lkeys:
            layer = self.contacts[lk]
            p1ages = self.age[layer['p1']]
            p2ages = self.age[layer['p2']]
            contact_counts[lk] = np.histogram(p1ages, edges)[0] + np.histogram(p2ages, edges)[0]

        # Plot contacts
        layer_colors = sc.gridcolors(n_layers)
        share_ax = None
        for w,w_type in enumerate(['total', 'percapita', 'weighted']): # Plot contacts in different ways
            for i,lk in enumerate(lkeys):
                if w_type == 'total':
                    weight = 1
                    total_contacts = 2*len(self.contacts[lk]) # x2 since each contact is undirected
                    ylabel = 'Number of contacts'
                    title = f'Total contacts for layer "{lk}": {total_contacts:n}'
                elif w_type == 'percapita':
                    weight = np.divide(1.0, age_counts, where=age_counts>0)
                    mean_contacts = 2*len(self.contacts[lk])/len(self) # Factor of 2 since edges are bi-directional
                    ylabel = 'Per capita number of contacts'
                    title = f'Mean contacts for layer "{lk}": {mean_contacts:0.2f}'
                elif w_type == 'weighted':
                    try:
                        weight = self.pars['beta_layer'][lk]*self.pars['beta']
                    except:
                        weight = 1
                    total_weight = np.round(weight*2*len(self.contacts[lk]))
                    ylabel = 'Weighted number of contacts'
                    title = f'Total weight for layer "{lk}": {total_weight:n}'

                ax = pl.subplot(n_rows, n_layers, n_layers*(w+1)+i+1, sharey=share_ax)
                pl.bar(bins, contact_counts[lk]*weight, color=layer_colors[i], width=width, zorder=zorder, alpha=alpha)
                pl.xlim([min_age-offset,max_age+offset])
                pl.xticks(np.arange(0, max_age+1, gridspace))
                pl.grid(True)
                pl.xlabel('Age')
                pl.ylabel(ylabel)
                pl.title(title)
                if w_type == 'weighted':
                    share_ax = ax # Update shared axis

        return fig


    def plot_graph(self):
        '''
        Convert to networkx and draw. WARNING: extremely slow for more than ~100 people!

        **Example**::

            pop = sp.Pop(n=50)
            pop.to_people().plot_graph()
        '''
        G, nodes, edges = self.to_graph(full_output=True)
        node_colors = [n['age'] for i,n in nodes]
        layer_map = dict(h='#37b', s='#e11', w='#4a4', c='#a49')
        edge_colors = [layer_map[G[i][j][k]['layer']] for i,j,k in edges]
        edge_weights = [G[i][j][k]['beta']*5 for i,j,k in edges]
        fig = pl.figure()
        nx.draw(G, node_color=node_colors, edge_color=edge_colors, width=edge_weights, alpha=0.5)
        return fig


    def story(self, uid, *args):
        '''
        Print out a short history of events in the life of the specified individual.

        Args:
            uid (int/list): the person or people whose story is being regaled
            args (list): these people will tell their stories too

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0)
            sim.run()
            sim.people.story(12)
            sim.people.story(795)
        '''

        def label_lkey(lkey):
            ''' Friendly name for common layer keys '''
            if lkey.lower() == 'a':
                llabel = 'default contact'
            if lkey.lower() == 'h':
                llabel = 'household'
            elif lkey.lower() == 's':
                llabel = 'school'
            elif lkey.lower() == 'w':
                llabel = 'workplace'
            elif lkey.lower() == 'c':
                llabel = 'community'
            else:
                llabel = f'"{lkey}"'
            return llabel

        uids = sc.promotetolist(uid)
        uids.extend(args)

        for uid in uids:

            p = self[uid]
            sex = 'female' if p.sex == 0 else 'male'

            intro = f'\nThis is the story of {uid}, a {p.age:.0f} year old {sex}'
            print(intro)

            total_contacts = 0
            no_contacts = []
            for lkey in p.contacts.keys():
                llabel = label_lkey(lkey)
                n_contacts = len(p.contacts[lkey])
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                nc_string = ', '.join(no_contacts)
                print(f'{uid} has no contacts in the {nc_string} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

        return