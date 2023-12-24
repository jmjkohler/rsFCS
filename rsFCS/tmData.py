import numpy as np

class tmData(object):
    """
    A superclass for time-mode FFS data  
    input variables:
    channels : a python list of channels in FFS data.
    frequency: an integer of frequency of FFS data collection (Hz)
    memorize : a flag variable which enables to memorize rebinned data
    """

    def __init__(self, channels=[], frequency=1, memorize=True):
        self._data = {}
        self._memorize = memorize
        self._filename = {}
        if self._checkchanneltype(channels):
            #make sure channels is a list or tuple before assigning
            self._channels = [i for i in channels]

        if isinstance(frequency, (int, float)):
            self._frequency = frequency
        else:
            raise TypeError("frequency must be integer or float, not {}".format(
                                type(frequency).__name__))


    def setData(self, channel, data):
        """
            set photon count data for the channel (channel)

            For example,
              td.setData(1, k) # set the data k for the channel 1

            To assign a list of multiple data, use
              td.data = [k1, k2, k3] # set the list of three sequences
        """
        if not isinstance(data, (np.ndarray)):
            raise TypeError("The type of data must be numpy.ndarray.")
        if channel not in self._channels:
            print("The channel must be in {}".format(self._channels))
            raise ValueError("Invalid channel number")
        self._data[channel] = {1: data}

    def getData(self, channel=None, lbin=1):
        if channel is None:
            for i in sorted(self._channels):
                return self.rebin(i, lbin)
        return self.rebin(channel, lbin)

    def rebin(self, channel=1, lbin=1):
        """
        Rebin a photon count sequence of channel ch by a rebin factor lbin

        k = td.rebin(1, 2) # k is a sequence of channel 1 with a rebin factor 2

        """
        self._checkchalbin(channel, lbin)
        k = self._data[channel].get(lbin, None)
        if k is None:
            k = self._newRebin(channel, lbin)
            if self._memorize:
                self._data[channel][lbin] = k
        return k

    # Get a frequency after the rebin with lbin
    def getFreq(self, lbin=1):
        return self._frequency*1./lbin

    # Get sampling time after rebin with lbin
    def getTSampling(self, lbin=1):
        return 1./self.getFreq(lbin)

    def info(self):
        """
        Returns the information in the class
        """
        return  {key:value for key, value in self.__dict__.items()
                         if not key.startswith('__') and not callable(key)}

    def reset(self):
        if self._data:
            self._data = { key: {1: d[1]} for key, d in self._data.items()}


    def _newRebin(self, channel, lbin):
        for i in sorted(self._data[channel].keys(), reverse=True):
            if lbin % i == 0:
                temp_ds = self._data[channel][i].size
                temp_lbin = lbin // i
                bds = temp_ds // (temp_lbin)
                k = (self._data[channel][i][:bds*temp_lbin]
                            .reshape(bds, temp_lbin).sum(axis=1))
                return k

    def _checkdata(self):
        if self.nchannels > 1:
            length = self._data[self._channels[0]][1].size
            for data in self._data.values():
                assert data[1].size == length, \
                    "The number of data points does not match."

    def _checkchalbin(self, ch, lbin):
        if ch not in self._channels:
            raise ValueError("Channel {} is not available".format(ch))
        if not isinstance(lbin, (int, np.int64) ):
            raise TypeError('lbin must be int, not {}'.format(type(lbin).__name__))

    def _checkchanneltype(self, channels):
        if isinstance(channels, (list, tuple)):
            if not self._data or len(channels)==len(self._data):
                return True
            else:
                raise ValueError("Number of channels assigned does not match\
 number of channels in data")
        else:
            raise TypeError("channels must be a python list, not {}".format(
                                type(channels).__name__))


    def __str__(self):
        l = ["{0}  :   {1}\n".format(key, value) for key, value in self.info().items()]
        return "".join(l)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, fname):
        self._filename = fname

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        if self._checkchanneltype(channels):
            self._channels = [i for i in channels]
            if self._data:
                newdatadict = {i:d for i,d in zip(self._channels, self._data.values())}
                self._data = newdatadict

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter#make read-only?
    def frequency(self, frequency):
        # reset the self._frequency and assign a new frequency
        if isinstance(frequency, (int, float)):
            self._frequency = frequency
        else:
            raise TypeError("frequency must be int or float")

    @property
    def data(self):
        # get the original data only.
        return { key: d[1] for key, d in self._data.items() }

    @data.setter
    def data(self, data):
        if not isinstance(data, list):
            raise TypeError("data must be a list.")
        if len(data) != self.nchannels:
            str0 = "The number of data channels in the list must be {}".format(self.nchannels)
            raise ValueError(str0)
        self._data = { i: {1:d} for i, d in zip(self._channels, data) }
        return

    @property
    def nchannels(self):
        return len(self._channels)

    @property
    def tsampling(self):
        return 1./self._frequency
