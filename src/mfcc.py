import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from bob import ap


class FeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def plot(signal):
        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal)
        plt.show()

    @staticmethod
    def mfcc(path):
        rate, signal = scipy.io.wavfile.read(str(path))
        # values to mfcc function
        win_length_ms = 20  # The window length of the cepstral analysis in milliseconds
        win_shift_ms = 10  # The window shift of the cepstral analysis in milliseconds
        n_filters = 24  # The number of filter bands
        n_ceps = 20  # The number of cepstral coefficients
        f_min = 0.  # The minimal frequency of the filter bank
        f_max = 8000.  # The maximal frequency of the filter bank
        delta_win = 2  # The integer delta value used for computing the first and second order derivatives
        pre_emphasis_coef = 0.97  # The coefficient used for the pre-emphasis
        dct_norm = True  # A factor by which the cepstral coefficients are multiplied
        mel_scale = True  # Tell whether cepstral features are extracted on a linear (LFCC) or Mel (MFCC) scale

        c = ap.Ceps(rate, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min, f_max, delta_win, pre_emphasis_coef,
                    mel_scale, dct_norm)
        csignal = np.cast['float'](signal)  # vector should be in **float**
        mfcc = c(csignal)
        mfcc = np.delete(mfcc, 0, 1)

        plt.figure(1)
        plt.title('Signal Wave...')
        plt.plot(signal)
        plt.show()
        return mfcc


'''
# Plot the features
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(q_mfcc, q_mfcc, q_mfcc)
'''
