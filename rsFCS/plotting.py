import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import scipy.stats as stats

def sciformatter():
    formatter = mpl.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    return formatter

def plot_acf(acfdict, corr_channel, fitdict=None, fig=None):
    if not fig:
        fig = plt.figure()
    if fitdict == None:
        ax1 = fig.add_subplot(111)
        axs = np.array([ax1])
    else:
        if corr_channel != fitdict['corr_channel']:
            warnings.warn("Specified channel does not match fitdict channel")
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05, height_ratios=[2,1])
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(fitdict['t'], fitdict['modelvals'], c='r', zorder=9)
        ax1.set_xscale('log')
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(fitdict['t'], fitdict['resids'], c='k')
        ax2.axhline(y=0, c='gray', ls='--')
        ax2.set_ylabel('residual')
        ax2.set_xlabel(r'$\tau$ (s)')
        [label.set_visible(False) for label in ax1.get_xticklabels()]
        axs = np.array([ax1, ax2])
    ax1.errorbar(acfdict['t'], acfdict[corr_channel]['Cor'],
            yerr=acfdict[corr_channel]['Corerr'], fmt='o')
    ax1.set_xscale('log')
    ax1.yaxis.set_major_formatter(sciformatter())
    ax1.set_ylabel(r'G($\tau$)')
    return axs

def plot_trace(tmdata, channels, ax=None, targetfreq=10):
    '''
    plot intensity trace of FFS data

    Parameters
    ----------
    tmdata : tmdata
        tmdata object containing FFS data to be plotted.
    channels : list
        channel(s) of the data to plot. Must be a list or iterable
    ax : matplotlib axes object
        Axes in which to plot the intensity. If None, a new figure and axes are
        created (default).
    targetfreq : int, default : 10
        Approximate frequency at which the intensity trace is displayed (Hz)

    Returns
    -------
    ax
        matplotlib axes object where the data was plotted

    '''
    if not isinstance(channels, list):
        raise TypeError("channels must be list")
    rebinfactor = tmdata.frequency//targetfreq #rebinfactor needed to get ~targetfreq Hz data for display
    newfreq = tmdata.getFreq(rebinfactor)
    if not ax:
        dfs = mpl.rcParams['figure.figsize']#default figure size
        fig = plt.figure(figsize=(dfs[1]*1.6, dfs[1]*0.6))
        ax = fig.add_subplot(111)
    for channel in channels:
        data = tmdata.rebin(channel=channel, lbin=rebinfactor)
        tvals = np.arange(0,len(data)/newfreq,1/newfreq)
        ax.plot(tvals, data/1000*newfreq)
    minval, maxval = _plot_data_range(ax)
    # maxval = np.max(data)*newfreq
    ax.set_ylim(0, 1.2*maxval)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Intensity (kcps)')
    fig = ax.figure
    fig.set_tight_layout(True)
    return ax

def _plot_data_range(ax):
    '''
    get min and max data y-values in plot from axes
    '''
    datalist = [ax.lines[i].get_ydata() for i in range(len(ax.lines))]
    minval, maxval = np.min(datalist), np.max(datalist)
    return minval, maxval

def _figax(ax):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # else:
    #     fig = ax.get_figure()
    return ax

def standard_normal(min=-4, max=4, npoints=100):
    xx = np.linspace(min, max, npoints)
    distribution = (1/np.sqrt(2*np.pi))*np.exp(-(xx**2)/2)
    return xx, distribution

def reduced_chi2(dof, lowppf=.001, hippf=.999, npoints=100):
    xx = np.linspace(stats.chi2.ppf(lowppf,dof),
        stats.chi2.ppf(hippf,dof), npoints)
    reduced_xx = xx/dof
    distribution = stats.chi2.pdf(xx, dof)*dof
    return reduced_xx, distribution
