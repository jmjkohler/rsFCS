import numpy as np
import warnings
import fitting

def shuffle_G(gtau_arr,k1_arr,nsegs_per_trace,rng=None):
    if not rng:
        rng = np.random.default_rng()
    ss = rng.bit_generator._seed_seq
    ntaus = gtau_arr.shape[1]
    totaltraces, remainder = divmod(gtau_arr.shape[0],nsegs_per_trace)
    if remainder != 0:
        warnings.warn("gtau_arr not integer multiple of nsegs_per_trace." \
                        " Dropping last {:} segments from rsACF".format(remainder))
        gtau_arr = gtau_arr[:nsegs_per_trace*totaltraces,:]
    new_g_arr = np.zeros_like(gtau_arr)
    new_k1_arr = np.zeros_like(gtau_arr)
    for i in range(gtau_arr.shape[0]):
        indxxs = rng.integers(low=0,high=gtau_arr.shape[0],size=ntaus)
        for j in range(ntaus):
            new_g_arr[i,j] = gtau_arr[indxxs[j],j]
            new_k1_arr[i,j] = k1_arr[indxxs[j],j]

    reshaped_newg = np.reshape(new_g_arr,(-1,nsegs_per_trace,ntaus))
    reshaped_newg_mean = np.zeros((totaltraces,ntaus))
    reshaped_newg_sem = np.zeros((totaltraces,ntaus))
    rnk1 = np.reshape(new_k1_arr,(-1,nsegs_per_trace,ntaus))
    u_rnk1_mean = np.zeros((totaltraces,ntaus))
    for i in range(totaltraces):
        for j in range(ntaus):
            uniquehere,ucounts = np.unique(reshaped_newg[i,:,j],return_counts=True)
            reshaped_newg_mean[i,j] = np.mean(uniquehere)
            reshaped_newg_sem[i,j] = np.std(uniquehere)/np.sqrt(len(uniquehere))
            uniquek1,uniquk1counts = np.unique(rnk1[i,:,j],return_counts=True)
            u_rnk1_mean[i,j] = np.mean(uniquek1)

    new_acfres_dict = dict()
    for i in range(totaltraces):
        temp_dict = {'Cor':reshaped_newg_mean[i,:],'Corerr':reshaped_newg_sem[i,:],
                    'Cor_segs':reshaped_newg[i,:,:],'k1':u_rnk1_mean[i,:]}
        new_acfres_dict.update({i: temp_dict})

    return new_acfres_dict
