'''
Little script for saving contacts to disk for ingestion by covasim.
'''

import pylab as pl
import sciris as sc
import synthpops as sp

contacts = sp.make_contacts(options_args={'use_microstructure': True})

# fn = 'contacts_48797.obj'
# sc.saveobj(filename=fn,obj=contacts)

cdict = sc.odict(contacts)
keys = cdict[0]['contacts'].keys()
counts = {}
for key in keys:
    counts[key] = []

for c in cdict.values():
    for key in keys:
        count = len(c['contacts'][key])
        counts[key].append(count)


for k,key in enumerate(keys):
    pl.subplot(2,2,k+1)
    pl.hist(counts[key], bins=50)
    pl.title(key)