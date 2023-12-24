import inspect
import numpy as np
from tmData import tmData
from Gtau_estimators import logMTau, parasLogMTau
from plotting import plot_acf, plot_trace
import data_read_utils as dru
import fitting

class FFSData:
    """A class for analysis and visualization of FCS data."""
    readers = dru.readers
    fitfuncs = fitting.ACFFitter.fitfuncs

    def __init__(self, channels=[], frequency=1, memorize=True):
        self._acfdict = {}
        self._fitdict = {}
        self._seglength = None
        self._corr_channel = None
        self._tmdata = tmData(channels, frequency, memorize)

    @classmethod
    def read(cls, reader_name, channels, frequency, filename, **kwargs):
        """Read photon count data into a new instance of FFSData.
        Parameters
        ----------
        reader_name : str
            Name of reader to decode raw photon count data. Available readers
            can be viewed with print(FFSData.readers). Builtin options are
            'flex', 'npy', and 'sav'.
        channels : list
            A list of the channels in the data. For example, [1,2] for two channel
            data
        frequency : int or float
            Sampling frequency of the data (Hz)
        filename : str
            filename or filepath to load the data from
        **kwargs
            keyword arguments passed to reader function

        Returns
        -------
        FFSData(channels, frequency) : FFSData
            Instance of FFSData(channels, frequency) with data loaded from filename
        """
        reader = cls.readers.get_object(reader_name)
        instance = cls(channels, frequency)
        reader_params = inspect.signature(reader).parameters
        for param, paramval in zip(['channels', 'frequency'],[channels,frequency]):
            if param in reader_params:
                kwargs = {param: paramval, **kwargs}
        instance._tmdata.data = reader(filename, **kwargs)
        instance._tmdata.filename = filename
        return instance

    def analyze(self, seglength, corr_channel=None, paras=parasLogMTau,
                model='2DG', show_cf=False, show_intensity=False, **kwargs):
        '''Calculate correlation function and fit to the specified model

        Parameters
        ----------
        seglength : int
            Segment length to use for segmenting the data.
        corr_channel : tuple, optional
            A 2-tuple specifying the channels correlate. For data with channels = [1,2],
            (1,1) specifies autocorrelation of channel 1, (2,2) specifies autocorrelation
            of channel 2, and (1,2) specifies cross-correlation of channels 1 and 2. The
            autocorrelation of the last channel is calculated if corr_channel is not
            supplied.
        paras : dict, optional
            A dictionary of parameters for the multiple tau calculation passed
            to Gtau_estimators.logMTau
        model : str, optional
            The fit model to use for fitting. Default is '2DG'. Available models
            can be seen by calling 'print(FFSData.fitfuncs)'
        show_cf : bool, optional
            A flag to determine whether correlation function is plotted. Default
            is False
        show_intensity : bool, optional
            A flag to determine whether the fluorescence intensity trace is plotted.
            Default is False
        **kwargs
            keyword arguments passed to fitting.ffit()

        Returns
        -------
        out_ax_arr : np.ndarray
            Array containing the matplotlib axes objects from the plots
        '''
        corr_channel = self._getdefaultcorrchannel(corr_channel)
        self.calc_corr(seglength, corr_channel, paras)
        self.fit(model=model, **kwargs)
        out_ax_arr = np.full(2, None)
        if show_cf:
            axs = self.plot_cf()
            axs[0].set_title(str(corr_channel))
            out_ax_arr[0] = axs
        if show_intensity:
            ax2 = self.plot_intensity()
            out_ax_arr[1] = ax2
        return out_ax_arr

    def calc_corr(self, seglength, corr_channel=None, paras=parasLogMTau):
        """
        Calculate the correlation function of corr_channel with a segment length
        equal to seglength. The result is memorized by the instance of FFSData.

        Parameters
        ----------
        seglength : int
            Segment length to use for segmenting the data.
        corr_channel : tuple, optional
            A 2-tuple specifying the channels correlate. For data with channels = [1,2],
            (1,1) specifies autocorrelation of channel 1, (2,2) specifies autocorrelation
            of channel 2, and (1,2) specifies cross-correlation of channels 1 and 2. The
            autocorrelation of the last channel is calculated if corr_channel is not
            supplied.
        paras : dict, optional
            A dictionary of parameters for the multiple tau calculation passed
            to Gtau_estimators.logMTau
        """
        corr_channel = self._getdefaultcorrchannel(corr_channel)
        if (seglength != self._seglength) or (corr_channel != self._corr_channel):
            self._acfdict = logMTau(self._tmdata, seglength, corr_channel, 1, paras)
            self._acfdict.update({'corr_channel': corr_channel})
        self._seglength = seglength
        self._corr_channel = corr_channel

    def fit(self, model='2DG', **kwargs):
        fitter = fitting.ACFFitter(self._acfdict)
        self._fitdict = fitter.fit(model, **kwargs)

    def plot_cf(self, showfit=True):
        """
        Plot correlation function.

        Parameters
        ----------
        showfit : bool, optional
            A flag to determine whether to overplot fit with correlation function
            estimator. Default is True.

        Returns
        -------
        axs : np.ndarray
            Numpy array of axes objects where data is plotted
        """
        if not self._fitdict:
            fitdict = None
        else:
            fitdict = self._fitdict
        axs = plot_acf(self._acfdict, self._corr_channel, fitdict)
        return axs

    def plot_intensity(self, channels=None, ax=None, targetfreq=10):
        """
        Plot intensity trace from raw photon count data

        Parameters
        ----------
        channels : list, optional
            List of channels to plot. If not provided, all channels will be plotted
        ax : matplotlib.axes, optional
            Axis to plot the data in. If not provided, a new figure is created
        targetfreq : int, optional
            Target frequency to rebin the photon count data for display of the
            intensity trace. Default is 10 Hz.

        Returns
        -------
        ax : matplotlib.axes
            Axis in which the intensity trace(s) were plotted.
        """
        if not channels:
            channels = self.channels
        axh = plot_trace(self._tmdata, channels, ax, targetfreq)
        return axh

    @property
    def frequency(self):
        return self._tmdata.frequency

    @property
    def channels(self):
        return self._tmdata.channels

    @property
    def filename(self):
        return self._tmdata.filename

    def corr_info(self):
        if not self._acfdict:
            outstr = 'No correlation function results availble'
        else:
            lines = ["{:<14}{:<12}{:<12}".format('seglength','tseg','corr_channel')]
            lines.append('-'*40)
            lines.append("{:<14}{:<12}{:<12}".format(self._seglength,
                self._seglength*self._tmdata.tsampling,str(self._corr_channel)))
            outstr = "\n".join(lines)
        return outstr

    def fit_info(self):
        if not self._fitdict:
            outstr = 'No fits available'
        else:
            lines = ["fit model: "+str(self._fitdict['modelname'])]
            lines.append('optimal parameters found:')
            for index, item in enumerate(self._fitdict['param_names']):
                lb = ("{:>9}"+": "+"{:<.2e}")
                nums = [item, self._fitdict['popt'][index]]
                newline = lb.format(*nums)
                lines.append(newline)
            outstr = "\n".join(lines)
        return outstr

    def __str__(self):
        lines = ['FFSData(channels={}, frequency={})'.format(self._tmdata.channels,
                            self._tmdata.frequency)]
        lines.append('filename: {}'.format(self.filename))
        lines.append('\n')
        lines.append('Correlation Function:')
        lines.append(self.corr_info())
        lines.append('\n')
        lines.append('Fits:')
        lines.append(self.fit_info())
        return "\n".join(lines)

    def getACF(self):
        return self._acfdict

    def getTau(self):
        '''convenience method to get lag times of ACF'''
        return self._acfdict['t']

    def getGtau(self):
        '''convenience method to get correlation function values'''
        return self._acfdict[self._corr_channel]['Cor']

    def getGtauerr(self):
        '''convenience method to get error bars on ACF'''
        return self._acfdict[self._corr_channel]['Corerr']

    def getFit(self):
        return self._fitdict

    def _getdefaultcorrchannel(self,corr_channel):
        if not corr_channel:
            return tuple((self.channels[-1],self.channels[-1]))
        else:
            if isinstance(corr_channel, tuple) and len(corr_channel)==2:
                return corr_channel
            else:
                raise TypeError('corr_channel must be a tuple of length 2')

    @classmethod
    def register_reader(cls, name, func):
        """
        Add a user-defined reader function for loading photon count data
        """
        cls.readers.register_object(name, func)

    @classmethod
    def register_fitfunction(cls, name, func, default_p0=None, default_fixed={}):
        """
        Add a user-defined fit function for correlation function fitting
        """
        outdict = dict()
        if not default_p0:
            p_names = [name for name in inspect.signature(func(None)).parameters]
            n_params = len(p_names[1:])
            default_p0 = [1]*n_params
        outdict.update({'func':func,'defaults':
                                    {'p0':default_p0,'fixed':default_fixed}})
        cls.fitfuncs.register_object(name, outdict)
