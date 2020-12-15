import synthpops as sp
import sciris as sc
import os

n = 5000
f = os.path.join(sp.datadir, 'synthpop_{0}.pop'.format(n))

print(f)
pop = sc.loadobj(f)
print(pop)
print(type(pop))
print()
keys = pop.keys()

print(keys)

print(pop['popdict'])