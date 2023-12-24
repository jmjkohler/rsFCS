from rsFCS.FFSData import FFSData
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Demonstration of how to use FFSData to compute sACF estimator and fit it for
# a single photon count data.
###############################################################################

#dummy file name. Assume the photon counts are stored in numpy .npy file
filename = 'trace_0.npy'

#load FCS photon count data into a FFSdata object
#  reader_name specifies format. Currently 'npy' (numpy npy file) and 'flex' (8-
#     or 16-bit binary file as produced by correlator.com flex card) are options
#  channels is a python list of channels in the data (single channel in this case)
ffsobj = FFSData.read(reader_name='npy', channels=[1], frequency=100000,
        filename=filename)

#ACFs are computed and fitted to a model with 'analyze()'
#  seglength is the segment length to use in the sACF analysis
#  corr channel is a tuple of the channel to compute and fit.
#  model chooses the fit model. Default is 2D Gaussian
#  p0 is an initial guess of parameters passed to scipy.optimize.curve_fit.
#    For '2DG', the parameters are [G0, td]
ffsobj.analyze(seglength=32768*20, corr_channel=(1,1), model='2DG',p0=[.05,.01])

#ffsobj.getFit() returns a python dictionary with results of the fit. Optimal
#    parameters are under keyword 'popt'. Red. chi2 value is under 'rcs'. Runs
#    test statistic is under 'runs_test_statistic'
fitresult = ffsobj.getFit()

#lag times can be conveniently accessed with .getTau()
taus = ffsobj.getTau()

#sACF values and errorbars are returned by .getGtau() and .getGtauerr(), respectively
gtau_vals = ffsobj.getGtau()
gtau_errs = ffsobj.getGtauerr()
