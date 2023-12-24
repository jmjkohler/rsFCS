from rsFCS.rsACF import rsACF
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Demonstration of how to use rsACF class to calculate rsACF estimators and fits.
# This example loads a series of photon count records saved as numpy .npy.
###############################################################################


# dummy file names. This example assumes data is saved in numpy .npy format
filenameslist = ['trace_'+str(i)+'.npy' for i in range(10)]

#initialize rsACF object
#    channels is a list of channels (single channel in this case)
#    frequency is the sampling frequency of the data
rsobj = rsACF(channels=[1], frequency=100000)

#load photon count record and compute autocorrelation
rsobj.batchload(filenamelist=filenameslist, reader_name='npy', seglength=40960,
                corr_channel=(1,1))

#randomize data to create and fit rsACF estimators
#  p0 is the initial guess for parameter values ([g0, td])
rsobj.analyze(model='2DG', p0=[.05,.01])

#a python dictionary with results of fitting is returned by 'get_rsACF_result'
results = rsobj.get_rsACF_result()

#display the reduced chi2 values from rsACF fitting
print(results['rcs'])
