import sciris as sc

def rehydrate(data):
    popdict = sc.dcp(data['popdict'])
    mapping = {'H':'households', 'S':'schools', 'W':'workplaces'}
    for key,label in mapping.items():
        for r in data[label]: # House, school etc
            for uid in r:
                for juid in r:
                    if uid != juid:
                        popdict[uid]['contacts'][key].add(juid)
    return popdict


sc.tic()
fn = '../data/synthpop_5000.pop'
data = sc.loadobj(fn)
sc.toc()
popdict = rehydrate(data)
sc.toc()