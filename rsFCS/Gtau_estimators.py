import numpy as np

"""Collection of FCS estimators


    logMTau
        calculate correlation estimator between two photon count data.

"""

def helper_logMTau(arr1, arr2, indices, segmentlength, reshaped_data):
    if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
        raise TypeError('arr1 and arr2 must be a numpy array.')

    n_segment = arr1.size // int(segmentlength)

    x1, x2 = indices
    if x1 not in reshaped_data:
    #     temp_arr1 = reshaped_data[x1]
    # else:
        temp_arr1 = arr1[0:segmentlength*n_segment]  \
                        .reshape((n_segment, segmentlength))
        reshaped_data[x1] = temp_arr1

    if x2 not in reshaped_data:
    #     temp_arr2 = reshaped_data[x2]
    # else:
        temp_arr2 = arr2[0:segmentlength*n_segment]  \
                        .reshape((n_segment, segmentlength))
        reshaped_data[x2] = temp_arr2

    return reshaped_data[x1], reshaped_data[x2]

parasLogMTau = {
    'tau1_min'   : 1,     # unrebinned data g(tau1_min : tau1_max)
    'tau1_max'   : 16,
    'binfactor'  : 4,    # rebin data consecutively by this factor
    'taur_min'   : 4,     # rebin data gr(taur_min : taur_max)
    'taur_max'   : 16,
    'npts_limit' : 16 #64
}


def helper_logMTau_GenerateTimeIndex(segmentlength, paras=parasLogMTau):
    time_index = [np.arange(paras['tau1_min'], paras['tau1_max'] )]

    bin_index = [np.full(time_index[0].size, 1)]
    number_of_rebins = (np.log(segmentlength/paras['npts_limit'])      \
                            / np.log(paras['binfactor']) + 1).astype(int)
    good_index = [[segmentlength-j>1 for j in range(paras['tau1_min'],
                paras['tau1_max'])]]
    for i in range(1, number_of_rebins):
        time_index.append(np.arange(paras['taur_min'], paras['taur_max'] )   \
                                    * paras['binfactor']**i)
        bin_index.append(np.full(time_index[i].size, paras['binfactor']**i))
        nsl = segmentlength//(paras['binfactor']**i)#new
        good_index.append([nsl-j>1 for j in range(paras['taur_min'],
                paras['taur_max'])])
    return np.concatenate(time_index), np.concatenate(bin_index), np.concatenate(good_index)



def logMTau(tmdata, segmentlength, corr_channel, lbin=1, paras=parasLogMTau):

    """logMtau function

    Parameters
    ----------
        tmdata : tmData
            abstraction of photon count data
        segmentlength : int
            segment length for the calculation of FCS estimators
        corr_channel : tuple
            tuple of channels to be correlated. (1,1) for autocorrelation of channel
            1, (1,2) for cross-correlation of channels 1 and 2, etc.
        lbin : int , optional
            rebin factor of the raw data to calculate the correlation function
            from. lbin must be in tmdata. Default 1.
        paras : dict, optional
            A dictionary of parameters to control the correlation function calculation.

    Returns
    -------
        result : dict
            collection of auto- and cross- correlations with their errors
    """

    time_indices, bin_indices, good_index = helper_logMTau_GenerateTimeIndex(segmentlength, parasLogMTau)
    result = {}
    time = time_indices / (tmdata.getFreq(lbin))
    result['t'] = time[good_index]
    result['b'] = bin_indices[good_index]
    result['k1'] = np.zeros_like(time, dtype=np.float64)  # this is necessary for the bias correction
    result['seglength'] = segmentlength
    result['frequency'] = tmdata.getFreq(lbin)
    result['tsampling'] = tmdata.getTSampling(lbin)
    result['tseg'] = result['tsampling']*segmentlength#test these
    result['lbin'] = lbin

    reshaped_data={} # collection of reshaped data to avoid
    # repeating the reshaping of photon count data

    arr1 = tmdata.getData(corr_channel[0], lbin=lbin)
    arr2 = tmdata.getData(corr_channel[1], lbin=lbin)

    temp_arr1, temp_arr2 = helper_logMTau(arr1, arr2, corr_channel, segmentlength, reshaped_data)
    temp_shape = temp_arr1.shape

    correlations = np.zeros(time_indices.size) #[]
    correlations_segs = np.zeros((temp_shape[0], time_indices.size))
    correlations_stderr = np.zeros(time_indices.size)#[]
    # x1, x2 = x
    x1, x2 = corr_channel[0], corr_channel[1]
    index = 0
    for t0 in range(paras['tau1_min'], paras['tau1_max']):  #+1
        t_arr1 = temp_arr1[:, 0:segmentlength-t0]
        t_arr2 = temp_arr2[:, t0:segmentlength]
        m0_1 = t_arr1.mean(axis=1)
        m0_2 = t_arr2.mean(axis=1)
        t_corr = (t_arr1*t_arr2).mean(axis=1)/(m0_1*m0_2) - 1.
        # ignore nan values
        indices = np.isfinite(t_corr)
        result['k1'][index] = np.mean((m0_1 + m0_2)[indices])/2/bin_indices[index]

        t_corr0 = t_corr[indices]
        correlations[index] = t_corr0.mean()
        correlations_segs[:, index] = t_corr
        correlations_stderr[index]   \
                = t_corr0.std(ddof=1) / np.sqrt(len(t_corr0))

        index += 1

    temp_segmentlength = segmentlength
    number_of_rebins = (np.log(segmentlength / paras['npts_limit'])  \
                    / np.log(paras['binfactor']) + 1).astype(int)
    for rbin in range(number_of_rebins - 1):
        temp_segmentlength = temp_segmentlength // paras['binfactor']
        y1, y2 = (x1, rbin), (x2, rbin)
        if y1 in reshaped_data:
            tt_arr1 = reshaped_data[y1]
        else:
            tt_arr1 = temp_arr1.reshape(temp_shape[0], temp_segmentlength,
                    temp_arr1.shape[1]//temp_segmentlength).sum(axis=2)
            reshaped_data[y1] = tt_arr1
        if y2 in reshaped_data:
            tt_arr2 = reshaped_data[y2]
        else:
            tt_arr2 = temp_arr2.reshape(temp_shape[0], temp_segmentlength,
                    temp_arr2.shape[1]//temp_segmentlength).sum(axis=2)
            reshaped_data[y2] = tt_arr2

        for t1 in range(paras['taur_min'], paras['taur_max']): # +1
            t_arr1 = tt_arr1[:, 0:temp_segmentlength - t1]
            t_arr2 = tt_arr2[:, t1:temp_segmentlength]
            m0_1 = t_arr1.mean(axis=1)
            m0_2 = t_arr2.mean(axis=1)
            t_corr = (t_arr1*t_arr2).mean(axis=1)/(m0_1*m0_2) - 1.
            # ignore nan values
            indices = np.isfinite(t_corr)
            result['k1'][index] = np.mean((m0_1 + m0_2)[indices])/2/bin_indices[index]

            t_corr0 = t_corr[indices]
            correlations[index] = t_corr0.mean()
            correlations_segs[:, index] = t_corr
            correlations_stderr[index]  = t_corr0.std(ddof=1)      \
                                / np.sqrt(len(t_corr0))
            index += 1
    result['k1'] = result['k1'][good_index]
    result[corr_channel] = {'Cor':correlations[good_index],
                 'Corerr' : correlations_stderr[good_index],
                 'Cor_segs' : correlations_segs[:,good_index]
                }

    del reshaped_data  #delete reshaped_data
    return result
