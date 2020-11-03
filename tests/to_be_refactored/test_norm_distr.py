import numpy as np
import synthpops as sp


if __name__ == '__main__':
    
    dic = dict()
    dic[0] = 0
    dic[1] = 0
    dic[2] = 0

    distr = dic
    # distr = sp.norm_dic(dic)
    print(distr)
    sorted_keys = sorted(distr.keys())
    print(sorted_keys)

    sorted_distr = np.array([distr[k] for k in sorted_keys], dtype=float)
    print(sorted_distr)

    norm_sorted_distr = np.maximum(0,sorted_distr) # Don't allow negatives- masks negative values to 0
    print(norm_sorted_distr,'n')

    if np.sum(norm_sorted_distr) > 0:
        norm_sorted_distr = norm_sorted_distr/norm_sorted_distr.sum() # Ensure it sums to 1
    print(norm_sorted_distr)

    n = np.random.multinomial(1, norm_sorted_distr, size=1)[0]
    print(n)

    ### synthpops method ###
    print(sp.sample_single(dic))

    a = [0, 4, 0]
    a = np.array(a, dtype=float)
    print(sp.sample_single(a))
