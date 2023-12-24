import numpy as np

def _checkx(x):

    if isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("The type of x is not correct.")


def Gtau_2dg(x, paras, tsampling=1):
    """Autocorrelation function
        G(tau) of a single diffusing species in 2DG PSF

    paras : [G0, td, offset]
    """

    if len(paras) != 3:
        raise ValueError("The number of elements in paras should be 3")

    if tsampling == 1:
        t = _checkx(x)
    else:
        t = _checkx(x)*tsampling

    return paras[0]/(1. + t/paras[1]) + paras[2]

def bin2_2DG(x, taud):
    """B2(t; taud) : the second order binning function

    Parameters
    ----------
        x : float-like
            sampling time T
        taud : float-like
            diffusion time $\\tau d$
    Returns
    -------
        B2(t; taud) : float-like
            2nd order binning function value
    References
    -------
        [1] J. Müller. 2004. Cumulant Analysis in Fluorescence Fluctuation Spectroscopy,
            Biophysical Journal 86:3981-3992, :doi:`10.1529/biophysj.103.037887`
    """

    t = x/taud
    return taud*taud*2.*(-1.*t + (1. + t)*np.log(1. + t))

def cor2_under(t, td, tsampling=1):
    """ Gtau function correcting the undersampling for 2DG
        t         >> lag time
        td        >> diffusion time
        tsampling >> sampling times to correct the undersampling effect
     """
    tt = t/td
    nt = tsampling/td
    cor = -2.*(1. + tt)*np.log(1. + tt) + (1. - nt + tt)*np.log(1. - nt + tt) + (1. + nt + tt)*np.log(1. + nt + tt)
    return cor/nt/nt


def Gtau_2DG_UnderBias(t, paras, k1=None, tseg=1., tsampling=1., bins=None):
    """
    Autocorrelation corrected for undersampling and segment length bias for a
    single diffusing species with a 2D Gaussian PSF

    Parameters
    ----------
        t : array-like
            an array of time-lags
        paras : array-like
            an array of parameters [G0, td, offset]
        k1 : array-like
            array of mean photon counts
        tseg : float
            segment time
        tsampling : float
            sampling time
        b : array-like
            an array of bin factors used for the mulitau implementation

    Returns
    -------
        G(t) : array-like
            G(t) or Auto-correlation function with under-sampling and bias
            corrections

    References
    -------
        [1] J. Kohler, K.-H. Hur, J. Mueller. 2023. Autocorrelation function of
            finite-length data in fluorescence correlation spectroscopy. Biophysical
            Journal 122:241-253, :doi:`10.1016/j.bpj.2022.10.027`
    """

    G = paras[0]*cor2_under(t, paras[1], tsampling*bins)

    index = (t < tseg/2.)

    numerator = (tseg - 2.*t[index])*tsampling
    denominator = np.power((tseg - t[index]), 2)*k1[index]
    G[index] = G[index] - numerator/denominator

    G = G - paras[0]/2. * (bin2_2DG(tseg, paras[1])
                            + bin2_2DG( np.absolute(tseg - 2.*t), paras[1]) \
                            - 2.*bin2_2DG(t, paras[1])) / (tseg - t)/(tseg - t)
    return G + paras[2]

def Gtau_3dg(x, paras, tsampling=1):
    """Autocorrelation function
        G(tau) of a single diffusing species in 3DG PSF

    paras : [G0, td, r, offset]
    """
    if len(paras) != 4:
        raise ValueError("The number of elements in paras should be 4")

    t = _checkx(x)
    return paras[0]/((1+t/paras[1])*np.sqrt(1+t/((paras[2]**2)*paras[1]))) + paras[3]

def bin2_3DG(t, taud, r):
    """ Second order binning function for the 3D Gaussian PSF

    Parameters
    ----------
        t : float-like
            sampling time T
        taud : float-like
            diffusion time $\\tau d$
        r : float-like
            ratio of the PSF axial beam waist to its lateral beam waist
    Returns
    -------
        B2 : float-like
            2nd order binning function value
    References
    -------
        [1] J. Müller. 2004. Cumulant Analysis in Fluorescence Fluctuation Spectroscopy,
            Biophysical Journal 86:3981-3992, :doi:`10.1529/biophysj.103.037887`
    """
    s = np.sqrt(r**2 - 1)
    x = t/taud

    a = (4*r*taud**2)/s
    b = (r*s) - (s*np.sqrt(r**2 + x))
    c = -(1+x)*np.log(((r-s)*(s+np.sqrt((r**2)+x)))/(np.sqrt(1+x)))

    return a*(b+c)

def cor2_3dg_under(t, td, r, tsampling):
    '''
    Autocorrelation function corrected for undersampling with 3D Gaussian PSF

    Parameters
    ----------
        t : float
            lag time
        td : float
            diffusion time
        r : float
            ratio of PSF axial beam waist to lateral beam waist
        tsampling : float
            sampling time

    Returns
    -------
        ACF : float
            Autocorrelation function corrected for undersampling

    References
    -------
        [1] B. Wu and J. Müller. 2005. Time-Integrated Fluorescence Cumulant
            Analysis in Fluorescence Fluctuation Spectroscopy. Biophysical
            Journal 89:2721-2735, :doi:`10.1529/biophysj.105.063685`
    '''
    tt = t/td
    nt = tsampling/td
    s = np.sqrt(r**2 -1)

    a = (2*r)/(s*(nt**2))
    b = 2*s*np.sqrt((r**2)+tt)+2*(1+tt)*np.log(((r-s)*(s+np.sqrt((r**2)+tt)))/
            (np.sqrt(1+tt)))
    c = -1*s*np.sqrt((r**2)+tt+nt)-(1+tt+nt)*np.log(((r-s)*(s+np.sqrt((r**2)+tt+nt)))/
            np.sqrt(1+tt+nt))
    d = -1*s*np.sqrt((r**2)+tt-nt)-(1+tt-nt)*np.log(((r-s)*(s+np.sqrt((r**2)+tt-nt)))/
            np.sqrt(1+tt-nt))
    return a*(b+c+d)

def Gtau_3DG_UnderBias(t, paras, k1=None, tseg=1, tsampling=1, b=None):
    '''
    Autocorrelation corrected for undersampling and segment length bias for a
    single diffusing species with a 3D Gaussian PSF

    Parameters
    ----------
        t : np.ndarray
            an array of time lags
        paras : array-like
            an array of parameters [G0, td, r, offset]
        k1 : np.ndarray
            array of mean photon counts
        tseg : float
            segment time
        tsampling : float
            sampling time
        b : np.ndarray
            an array of bin factors used for the mulitau implementation

    Returns
    -------
        G(t) : np.ndarray
            G(t) or Auto-correlation function with undersampling and segment-length
            bias corrections

    References
    -------
        [1] J. Kohler, K.-H. Hur, J. Mueller. 2023. Autocorrelation function of
            finite-length data in fluorescence correlation spectroscopy. Biophysical
            Journal 122:241-253, :doi:`10.1016/j.bpj.2022.10.027`
    '''
    if k1 is None:
        raise ValueError("k1 not provided.")

    G = paras[0]*cor2_3dg_under(t, paras[1], paras[2], tsampling*b)

    index = (t < tseg/2.)

    numerator = (tseg - 2.*t[index])*tsampling
    denominator = np.power((tseg - t[index]), 2)*k1[index]
    G[index] = G[index] - numerator/denominator

    G = G - paras[0]/2. * (bin2_3DG(tseg, paras[1], paras[2])
                            + bin2_3DG( np.absolute(tseg - 2.*t), paras[1], paras[2]) \
                            - 2.*bin2_3DG(t, paras[1], paras[2])) / (tseg - t)/(tseg - t)
    return G + paras[3]

def Gtau_2DG_2DG_UnderBias(t, paras, k1=None, tseg=1, tsampling=1, b=None):
    n1, td1, γ1, n2, td2, γ2, offset = paras
    #γ factor for 2D Gaussian PSF is 0.5
    f1 = n1/(n1+n2)
    f2 = n2/(n1+n2)
    k_1 = f1*k1
    k_2 = f2*k1
    G1 = Gtau_2DG_UnderBias(t,[γ1/n1, td1, 0], k_1, tseg, tsampling, b)
    G2 = Gtau_2DG_UnderBias(t,[γ2/n2, td2, 0], k_2, tseg, tsampling, b)
    return (f1**2)*G1 + (f2**2)*G2 + offset


def Gtau_3DG_2DG_UnderBias(t, paras, k1=None, tseg=1, tsampling=1, b=None):
    n1, td1, γ1, r, n2, td2, γ2, offset = paras
    #γ factor for 3D Gaussian PSF is √2/4 ≈ 0.35
    #γ factor for 2D Gaussian PSF is 0.5
    f1 = n1/(n1+n2)
    f2 = n2/(n1+n2)
    k_1 = f1*k1
    k_2 = f2*k1
    G1 = Gtau_3DG_UnderBias(t, [γ1/n1, td1, r, 0], k_1, tseg, tsampling, b)
    G2 = Gtau_2DG_UnderBias(t, [γ2/n2, td2, 0], k_2, tseg, tsampling, b)
    return (f1**2)*G1 + (f2**2)*G2 + offset
