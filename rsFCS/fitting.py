import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from Gtau_models import Gtau_2DG_UnderBias, Gtau_2dg, Gtau_3DG_UnderBias, Gtau_3dg, Gtau_2DG_2DG_UnderBias, Gtau_3DG_2DG_UnderBias
import warnings, inspect
from functools import partial
from registry import Registry

def redchisquare(data,model,error,constraints):
    chisquare = np.sum(np.divide(np.square(data-model),np.square(error)))
    reduced = chisquare/(len(data)-constraints)
    return reduced

def resid(data,model,error):
    return np.divide(data-model,error)

def runstest(data):
    '''
    calculate the runs test statistic
    '''
    if not isinstance(data,np.ndarray) or data.ndim!=1:
        raise Exception('Data must be 1D numpy array')
    signs = np.sign(data)
    Nplus = len(signs[signs==1])
    Nminus = len(signs[signs==-1])
    Ntot = Nplus + Nminus
    if Ntot != len(data):
        warnings.warn("Nplus + Nminus does not equal number of data points")
    nrunscounter = 1
    for i in range(len(signs)-1):
        if signs[i+1] != signs[i]:
            nrunscounter += 1
    nruns = nrunscounter
    mu = ((2*Nplus*Nminus)/Ntot)+1
    sigmasquared = ((mu-1)*(mu-2))/(Ntot-1)
    sigma = np.sqrt(sigmasquared)
    Z = (nruns-mu)/sigma
    outdict = dict({'Nplus':Nplus,'Nminus':Nminus,'Ntot':Ntot,'nruns':nruns,
                    'mu':mu,'sigma':sigma,'test_statistic':Z})
    return outdict

def G2DG(t, g0, td, offset, k1, tseg, tsampling, b):
    return Gtau_2DG_UnderBias(t,[g0,td,offset],k1, tseg, tsampling,b)

def G2DG_inf(t, g0, td, offset):
    return Gtau_2dg(t, [g0, td, offset], tsampling=1)

def G3DG(t, g0, td, r, offset, k1, tseg, tsampling, b):
    return Gtau_3DG_UnderBias(t, [g0, td, r, offset], k1, tseg, tsampling, b)

def G3DG_inf(t, g0, td, r, offset):
    return Gtau_3dg(t, [g0, td, r, offset], tsampling=1)

def G2DG2DG(t, n1, td1, n2, Δtd, offset, k1, tseg, tsampling, b):
    γ2DG = 0.5
    return Gtau_2DG_2DG_UnderBias(t, [n1, td1, γ2DG, n2, td1+Δtd, γ2DG, offset], k1,
     tseg, tsampling, b)

def G3DG2DG(t, n1, td1, r, n2, Δtd, offset, k1, tseg, tsampling, b):
    γ2DG = 0.5
    γ3DG = 0.353553
    return Gtau_3DG_2DG_UnderBias(t,[n1, td1, γ3DG, r, n2, td1+Δtd, γ2DG, offset], k1,
     tseg, tsampling, b)

def ffit(func, x, y, fixed, **kwargs):
    ''' Non-linear least squares fitting of model function to data. This function
    wraps scipy.optimize's 'curve_fit' to allow parameters to be fixed in the fit

    Parameters
    ----------
    func : callable
        The model function. It must take the independent variable as its first argument.
    x : array_like
        The independent variable where measurements are taken.
    y : array_like
        The dependent data
    fixed : dict
        A dictionary specifying parameters to fix during fitting. The keys should
        match the function signature for the desired parameter, and the value is
        the numerical value at which to fix the parameter. For example, the function
        f(x,m,b) = m*x + b could hold the 'b' parameter constant at 0 by passing
        fixed = {'b':0}.
    **kwargs
        Keyword arguments passed to opt.curve_fit. p0 is the initial guess for
        the parameters and is mandatory. bounds gives upper and lower bounds on
        parameters. See curve_fit documentation for more information.

    Returns
    -------
    ff :
        Result of the fitting. ff[0] has the optimal parameters found, and ff[1]
        contains the estimated covariance of ff[0]
    modelvals : array
        Function evaluated at x with the optimal parameters found
    p_names : list
        List of parameter names left free during fitting

    '''
    pb = inspect.signature(func).bind_partial(**fixed)
    p_names = [name for name in inspect.signature(func).parameters
                if name not in pb.kwargs]
    func_fixed = partial(func, **pb.kwargs)
    def newfxn(*args, **nfkwargs):
        nfkwargs = {**{param: arg for param, arg in zip(p_names,args)},**nfkwargs}
        return func_fixed(**nfkwargs)
    ff = opt.curve_fit(newfxn, x, y, **kwargs)
    modelvals = newfxn(x, *ff[0])
    return ff, modelvals, p_names[1:]

def calc_fitstats(fit_result, modelvals, p_names, data, error):
    popt, pcov = fit_result[0], fit_result[1]
    perr = np.sqrt(np.diag(pcov))
    nparams = len(p_names)
    rcs = redchisquare(data, modelvals, error, nparams)
    resids = resid(data, modelvals, error)
    runsteststatistic = runstest(resids)['test_statistic']
    resultdict = dict({'popt': popt, 'pcov': pcov, 'perr': perr,
                     'modelvals': modelvals, 'rcs': rcs, 'resids': resids,
                     'runs_test_statistic': runsteststatistic,
                     'param_names':p_names,})
    if len(fit_result) == 5:
        resultdict.update({'infodict': fitresult[2], 'mesg': fitresult[3], 'ier':
        fitresult[4]})
    return resultdict

class ACFFitter:
    """A class for fitting autocorrelation functions"""
    BUILTIN_FITFUNCS = {'2DG' : {'func':G2DG,
                                'defaults': {'p0':[.005,.005],
                                            'fixed':{'offset':0}}},
                        '2DG_inf': {'func':G2DG_inf,
                                    'defaults': {'p0':[.005,.005],
                                                'fixed':{'offset':0}}},

                        '3DG' : {'func':G3DG,
                                'defaults': {'p0':[.005,.005],
                                            'fixed':{'r':3, 'offset':0}}},
                        '3DG_inf' : {'func':G3DG_inf,
                                    'defaults': {'p0':[.005,.005],
                                                'fixed':{'r':3,'offset':0}}},

                        '2DG2DG' : {'func':G2DG2DG,
                                                    'defaults': {'p0':[50,.001,50,.05],
                                                                'fixed':{'offset':0},
                                                                'bounds':(0,np.inf)}},
                        '3DG2DG' : {'func':G3DG2DG,
                                                    'defaults': {'p0':[50,.001,50,.05],
                                                                'fixed':{'r':2.11111,
                                                                       'offset':0},
                                                                'bounds':(0,np.inf)}}
                                            }
    fitfuncs = Registry()
    for name, func in BUILTIN_FITFUNCS.items():
        fitfuncs.register_object(name, func)

    def __init__(self, acfdict):
        self.acfdict = acfdict
        self.corr_channel = acfdict['corr_channel']

    def fit(self, model='2DG', **kwargs):
        fitfunc, defaults, segparamdict = self.makesegparamdict(model)
        fitkwargs = {**defaults, **kwargs}
        for key, value in segparamdict.items():
            fitkwargs['fixed'].update({key:value})

        res, modelvals, p_names = ffit(fitfunc, self.acfdict['t'],
                self.acfdict[self.corr_channel]['Cor'],
                sigma=self.acfdict[self.corr_channel]['Corerr'], absolute_sigma=True,
                **fitkwargs)
        summary = calc_fitstats(res, modelvals, p_names,
            self.acfdict[self.corr_channel]['Cor'], self.acfdict[self.corr_channel]['Corerr'])
        summary.update({'corr_channel':self.corr_channel,
                        'seglength':self.acfdict['seglength'],
            'modelname':model,'fit_kwargs':fitkwargs, 't':self.acfdict['t']})
        if 'lbin' in self.acfdict:
            summary.update({'lbin': self.acfdict['lbin']})
        return summary

    def makesegparamdict(self, model):
        fitfuncdict = ACFFitter.fitfuncs.get_object(model)
        fitfunc = fitfuncdict['func']
        defaults = fitfuncdict.get('defaults', {})#return defaults if defined, or empty dict if not defined
        segparamnames = ['k1','tseg','tsampling','b']
        funcsig = inspect.signature(fitfunc).parameters
        segparamstofix = [key for key in segparamnames if key in funcsig]
        segparamdict = dict()
        for key in segparamstofix:
            segparamdict.update({key:self.acfdict[key]})
        return fitfunc, defaults, segparamdict


    def print_fitmodels(self):
        """print the fit models currently available"""
        fms = ACFFitter.fitfuncs._objects.keys()
        for i in fms:
            print(i)
