import numpy as np
import matplotlib.pyplot as plt
from FFSData import FFSData
from random_sampling import shuffle_G
import warnings, os
from Gtau_estimators import parasLogMTau, helper_logMTau_GenerateTimeIndex
from concurrent.futures import ThreadPoolExecutor
import fitting, plotting

class rsACF:
    """A class for computing and fitting rsACF estimators"""
    def __init__(self, channels, frequency):
        self._channels = channels
        self._frequency = frequency
        self.seglength = None
        self.corr_channel = tuple()
        self.filenames = []

        self.gtau_arr = None
        self.k1_arr = None
        self.sACFdicts = {}
        self.nsegs_per_trace_set = set()
        self.nsegs_per_trace = None
        self.auxdict = None

        self.rsACFdicts = {}
        self.fitdict_rsACF = {} #randomized fits
        self.fitdict_sACF = {} #regular fits at same seglength
        self.fitresultarrs = {}

    def batchload(self, filenamelist, reader_name, seglength, corr_channel,
    paras=parasLogMTau, workers=5, **kwargs):
        '''Load a series of data traces concurrently'''
        for filename in filenamelist:
            if not os.path.exists(filename):
                raise FileNotFoundError("File not found: '{}'".format(filename))
        self.filenames = filenamelist
        self.corr_channel = corr_channel
        self.seglength = seglength
        def load_file(enumtup):
            index, item = enumtup
            data = FFSData.read(reader_name, self._channels, self._frequency,
                                item, **kwargs)
            data.calc_corr(seglength, corr_channel, paras)
            self.sACFdicts.update({index: data.getACF()})

        with ThreadPoolExecutor(workers) as executor:
            executor.map(load_file, enumerate(filenamelist))

        self.gtau_arr = np.vstack([self.sACFdicts[i][corr_channel]['Cor_segs']
            for i in sorted(self.sACFdicts.keys())])
        self.k1_arr = np.vstack([np.tile(self.sACFdicts[i]['k1'],
            (self.sACFdicts[i][corr_channel]['Cor_segs'].shape[0],1))
            for i in sorted(self.sACFdicts.keys())])
        nsegs_per_trace,ntaus_per_trace = zip(
            *[self.sACFdicts[i][corr_channel]['Cor_segs'].shape
            for i in sorted(self.sACFdicts)])
        self.nsegs_per_trace_set = set(nsegs_per_trace)
        self._fillauxdict()
        if len(self.nsegs_per_trace_set)>1:
            warnings.warn("Number of segments per trace is variable ("\
            "{:} segments per trace) ".format(self.nsegs_per_trace_set))
        self.nsegs_per_trace = min(self.nsegs_per_trace_set)

    def load_correlation_data(self, gtau_arr, k1_arr, auxdict=None):
        if not isinstance(gtau_arr, np.ndarray) or not isinstance(k1_arr, np.ndarray):
            raise TypeError("gtau_arr and k1_arr must be numpy arrays")
        if gtau_arr.shape != k1_arr.shape:
            raise ValueError("shapes of gtau_arr and k1_arr should be the same")
        if auxdict is not None:
            if not isinstance(auxdict, dict):
                raise TypeError ("auxdict must be dict")
            if not all([key in auxdict for key in ['t','b','seglength']]):
                raise ValueError("One or more of 't', 'b', 'seglength' missing from auxdict")
            if 'frequency' not in auxdict:
                auxdict.update({'frequency': self._frequency})
            if 'tsampling' not in auxdict:
                auxdict.update({'tsampling': 1/self._frequency})
            if 'tseg' not in auxdict:
                auxdict.update({'tseg': auxdict['seglength']/self._frequency})

            self.auxdict = auxdict
        if (len(self.auxdict['t']) != gtau_arr.shape[1]) or (len(self.auxdict['b']) !=
                                                             gtau_arr.shape[1]):
            raise ValueError("'t' or 'b' in auxdict incompatible with gtau_arr shape")

        self.gtau_arr = gtau_arr
        self.k1_arr = k1_arr
        if 'corr_channel' in self.auxdict:
            if not(isinstance(self.auxdict['corr_channel'],tuple) and
            len(self.auxdict['corr_channel'])==2):
                raise ValueError("'corr_channel should be tuple of length 2'")
            self.corr_channel = self.auxdict['corr_channel']
        else:
            self.corr_channel = (1,1)
            print('Setting corr_channel to (1,1)')


    def analyze(self, model='2DG', nsegs_per_trace=None, rng=None,
                summaryplot=False, savedir=None, **kwargs):
        '''Compute and fit rsACF estimators

        Parameters
        ----------
        model : str, optional
            Model function to fit correlation function estimators to. Default is
            '2DG'
        nsegs_per_trace : int, optional
            Number of segments to use per rsACF estimator. Default is the same
            as the minimum number of segments per trace in the raw data.
        rng : numpy.rng, optional
            Random number generator. If not supplied, a new rng will be initialized
        summaryplot : bool, optional
            Flag to control whether fit statistic histograms are displayed.
            Default is False
        savedir : path-like, optional
            Directory to save individual ACF plots to, if specified. If not
            specified, no ACF plots are saved (default)
        **kwargs
            keyword arguments passed to fitting.ffit()
        '''
        self._checkforacfdicts(nsegs_per_trace)
        self.fit(method='sACF', model=model, **kwargs)
        self.randomize(nsegs_per_trace, rng)
        self.fit(method='rsACF', model=model, **kwargs)
        if summaryplot:
            fig, axs = plt.subplots(2,3,figsize=(12,7))
            for row, method in enumerate(['rsACF','sACF']):
                for col, stat in enumerate(['rcs','resids','runs_test_statistic']):
                    self.plot_hist(ax=axs[row,col],method=method,stat=stat)
                    axs[row,col].set_title(method)
            plt.tight_layout()
        if savedir:
            self.save_acfplots(method='sACF',savedir=savedir)
            self.save_acfplots(method='rsACF',savedir=savedir)

    def randomize(self, nsegs_per_trace=None, rng=None):
        """Randomly sample ACF to create rsACF estimators"""
        if not nsegs_per_trace:
            nsegs_per_trace = self.nsegs_per_trace
        tempdict = shuffle_G(self.gtau_arr, self.k1_arr,
                        nsegs_per_trace, rng)
        for i in tempdict:
            self.rsACFdicts.update({i:dict()})
            for item in ['t','b','seglength','frequency','tsampling','tseg']:
                self.rsACFdicts[i].update({item: self.auxdict[item]})
            self.rsACFdicts[i].update({'k1': tempdict[i]['k1']})
            self.rsACFdicts[i].update({'corr_channel': self.corr_channel})
            self.rsACFdicts[i].update({self.corr_channel: dict()})
            self.rsACFdicts[i][self.corr_channel].update({'Cor':tempdict[i]['Cor'],
                'Corerr': tempdict[i]['Corerr'], 'Cor_segs': tempdict[i]['Cor_segs']})

    def fit(self, method='rsACF', model='2DG', **kwargs):
        """fit ACF to specified model function

        Parameters
        ----------
        method : str, optional
            Choose which analysis to plot. Choices are 'sACF' (non-randomized) or
            'rsACF' (randomly sampled ACF). Default is 'rsACF'
        model : str, optional
            Fit model to use in fitting. Default is '2DG'
        **kwargs
            keyword arguments passed to fitting.ffit
        """
        ACFs, fitresult = self._selectmethod(method)
        for i in sorted(ACFs):
            fitter = fitting.ACFFitter(ACFs[i])
            summary = fitter.fit(model, **kwargs)
            fitresult.update({i: summary})
        self.fitresultarrs.update({method:{}})
        poptarr = np.vstack([fitresult[i]['popt'] for i in fitresult])
        self.fitresultarrs[method].update({'popt': poptarr})
        keysforfra = ['rcs','resids','runs_test_statistic']
        for item in keysforfra:
            itemarr = np.vstack([fitresult[i][item] for i in fitresult]).flatten()
            self.fitresultarrs[method].update({item: itemarr})

    def save_acfplots(self, method='rsACF', savedir='.', **kwargs):
        """save plots of individual ACFs in savedir"""
        ACFs, fitresult = self._selectmethod(method)
        dirtosave = os.path.join(savedir, method)
        os.makedirs(dirtosave, exist_ok=True)
        for i in sorted(ACFs):
            axs = plotting.plot_acf(ACFs[i], self.corr_channel, fitresult[i])
            fig = axs[0].get_figure()
            plt.savefig(os.path.join(dirtosave,'acf_'+str(i).zfill(3)+'.png'),
                        bbox_inches='tight',**kwargs)
            plt.close(fig)

    def plot_hist(self, method='rsACF', stat='rcs', ax=None, **kwargs):
        """Plot histograms of fit statistics

        Parameters
        ----------
        method : str, optional
            Choose which analysis to plot. Choices are 'sACF' (non-randomized) or
            'rsACF' (randomly sampled ACF). Default is 'rsACF'
        stat : str, optional
            Which statistic to plot. Choices are 'rcs' (reduced chi square), 'resids'
            (residuals), or 'runs_test_statistic'. Default is 'rcs'
        ax : axes object, optional
            Matplotlib axes to plot the histogram in. If not provided, a new figure
            will be created
        **kwargs
            keyword arguments passed to plt.hist

        Returns
        -------
        ax : axes object
            axes histogram was plotted in
        """
        ACFs, fitresult = self._selectmethod(method)
        statvals = np.vstack([fitresult[i][stat] for i in fitresult]).flatten()
        ntaus = len(fitresult[0]['t'])
        nparams = len(fitresult[0]['param_names'])
        dof = ntaus-nparams
        xlabelsdict = {'rcs':r'reduced $\chi^2$',
                        'resids':'Standardized Residual',
                        'runs_test_statistic':'runs test statistic'}
        distsdict = {'rcs':plotting.reduced_chi2(dof),
                    'resids':plotting.standard_normal(),
                    'runs_test_statistic':plotting.standard_normal()}
        ax = plotting._figax(ax)
        ax.hist(statvals, bins='auto', density=True, **kwargs)
        ax.set_ylabel('PDF')
        ax.set_xlabel(xlabelsdict[stat])
        xdata, ydata = distsdict[stat]
        ax.plot(xdata, ydata)
        return ax

    def _add_acfdicts_from_gtauarr(self, nsegs_per_trace):
        self.nsegs_per_trace = nsegs_per_trace
        ntraces, remainder = divmod(self.gtau_arr.shape[0], nsegs_per_trace)
        if remainder != 0:
            warnings.warn("gtau_arr not integer multiple of nsegs_per_trace." \
                            " Dropping last {:} segments from sACF".format(remainder))
        corrchannel = self.corr_channel
        for i in range(ntraces):
            gshere = self.gtau_arr[i*nsegs_per_trace:(i+1)*nsegs_per_trace,:]
            cor = np.mean(gshere, axis=0)
            corerr = np.std(gshere, axis=0, ddof=1)/np.sqrt(nsegs_per_trace)
            k1here = self.k1_arr[i*nsegs_per_trace,:]
            cordict = {'Cor': cor, 'Corerr': corerr, 'Cor_segs': gshere}
            acfdicthere = {corrchannel:cordict}
            for key in ['t','b','seglength','frequency','tsampling','tseg']:
                acfdicthere.update({key: self.auxdict[key]})
            acfdicthere.update({'k1': k1here, 'corr_channel': corrchannel})
            self.sACFdicts.update({i: acfdicthere})

    def _selectmethod(self, method):
        if method not in ['rsACF','sACF']:
            raise ValueError("method should be 'rsACF' or 'sACF' ")
        dictselect = {'rsACF': self.rsACFdicts, 'sACF': self.sACFdicts}
        fitselect = {'rsACF':self.fitdict_rsACF, 'sACF': self.fitdict_sACF}
        ACFs = dictselect[method]
        fitresult = fitselect[method]
        return ACFs, fitresult

    def _checkforacfdicts(self, nsegs_per_trace):
        if len(self.sACFdicts.keys())==0:
            if not nsegs_per_trace:
                raise ValueError("nsegs_per_trace cannot be inferred and must be provided")
            self._add_acfdicts_from_gtauarr(nsegs_per_trace)

    def fit_info(self, method='rsACF'):
        if method not in ['rsACF','sACF']:
            raise ValueError("method must be either 'sACF' or 'rsACF'")
        if not self.fitresultarrs:
            outstr = 'No fits available'
        else:
            lines = [method]
            lines.append('summary of fit statistics:')
            lines.append("{:<16}{:<16}{:<14}".format('mean red. chi2', 'mean runs test', 'std resids'))
            lines.append('-'*40)
            rcsmean = np.mean(self.fitresultarrs[method]['rcs'])
            rtmean = np.mean(self.fitresultarrs[method]['runs_test_statistic'])
            residsstd = np.std(self.fitresultarrs[method]['resids'])
            lines.append("{:^16.2f}{:^16.2f}{:^14.2f}".format(rcsmean, rtmean, residsstd))
            lines.append("\n")
            lines.append('Parameter estimates: (mean,  SD of multiple fits)')
            meanparams = np.mean(self.fitresultarrs[method]['popt'],axis=0)
            stdparams = np.std(self.fitresultarrs[method]['popt'], axis=0)
            for index, name in enumerate(self.fitdict_rsACF[0]['param_names']):
                lines.append("{:<7}{:>8.2e} , {:>8.2e}".format(name,meanparams[index],stdparams[index]))
            outstr = "\n".join(lines)
        return outstr

    def _fillauxdict(self):
        if not self.auxdict:
            self.auxdict = dict()
        for item in ['t','b','seglength','frequency','tsampling','tseg']:
            self.auxdict.update({item: self.sACFdicts[0][item]})

    def gen_auxdict(self, seglength, paras=parasLogMTau):
        tt, bb, gg = helper_logMTau_GenerateTimeIndex(seglength)
        t2, b2 = tt[gg], bb[gg]
        ti = t2/self._frequency
        auxdicttemp = dict({'t': ti, 'b': b2, 'seglength':seglength, 'frequency':
                            self._frequency, 'tsampling':1/self._frequency,
                            'tseg': seglength/self._frequency})
        self.auxdict = auxdicttemp

    def __str__(self):
        lines = ['rsACF(channels={}, frequency={})'.format(self._channels,
                            self._frequency)]
        lines.append('number of rsACF fits: '+str(len(self.fitdict_rsACF)))
        if not self.sACFdicts and not self.gtau_arr:
            lines.append('No data loaded')
        lines.append(self.fit_info(method='rsACF'))
        return "\n".join(lines)

    @property
    def channels(self):
        return self._channels
    @property
    def frequency(self):
        return self._frequency

    def get_rsACF_result(self):
        return self.fitresultarrs['rsACF']
