import numpy as np
import scipy as sp 
from scipy.signal.windows import flattop
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt


def load_wav(file, normalize=False):
    """
    load wav and normalize to -1.0/+1.0 if noramlize=True
    returns fs, data
    """
    fs, data = sp.io.wavfile.read(file)
    
    # normalize audio to -1.0 to +1.0
    if normalize is not False:
        if np.issubdtype(data.dtype, np.integer):
            data = data / np.iinfo(data.dtype).max

    return fs, data

def thd_r(signal, fs, max_harmonic=19):
    # 1. apply hanning window - this attenuated by 0.5 which we will fix later
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # 2. perform FFT on the windowed signal
    yf = np.fft.rfft(signal_windowed)
    
    # 3. calculate magnitudes
    # we normalize by the window instead of the usual N and multiply by 2 for one-sided spectrum
    magnitudes = np.abs(yf) * 2 / np.sum(window)
    
    # 4. find the fundamental in the signal from the magnitudes, ie. not the 1st index (that would be DC) but the 2nd
    fundamental_idx = np.argmax(magnitudes[1:]) + 1
    fundamental_mag_peak = magnitudes[fundamental_idx]
    
    # 5. calculate frequency resolution
    freq_res = fs / len(signal)
    f1_freq = fundamental_idx * freq_res
    
    print(f"Detected Fundamental: {f1_freq:.2f} Hz")

    # calculate THD sums
    sum_squares_harmonics = 0
    sum_squares_total = fundamental_mag_peak ** 2
    
    # store the harmonic data
    harmonic_data = {} 
    
    # fundamental RMS value
    harmonic_data[1] = fundamental_mag_peak / np.sqrt(2)

    #go through harmonics up until max_harmonic
    for n in range(2, max_harmonic + 1):
        
        expected_idx = n * fundamental_idx
        search_range = 2 
        
        start = max(0, expected_idx - search_range)
        end = min(len(magnitudes), expected_idx + search_range + 1)
        
        # check for valid start (less than magnitudes array)
        if start < len(magnitudes):
            harm_mag_peak = np.max(magnitudes[start:end])
            
            sq_val = harm_mag_peak ** 2
            sum_squares_harmonics += sq_val
            sum_squares_total += sq_val
            
            # convert from peak to rms
            harmonic_data[n] = harm_mag_peak / np.sqrt(2)

    # compute THD_R 
    numerator = np.sqrt(sum_squares_harmonics)
    denominator = np.sqrt(sum_squares_total)
    
    # check for divide by zero 
    if denominator == 0:
        thd_r = 0
    else:
        thd_r = numerator / denominator

    return thd_r, harmonic_data

def timd(signal, fs, f_mod, f_carrier, n_max=3, search_window=5):
    # window = np.hanning(len(signal)) # Maybe use windowing 
    # window = np.blackman(len(signal))
    # window = np.kaiser(len(signal), 0)
    window = flattop(len(signal))
    signal_windowed = signal * window
    
    yf = np.fft.rfft(signal_windowed)
    # yf = np.fft.rfft(signal)
    magnitudes = np.abs(yf) * 2 / np.sum(window) # len(signal) # np.sum(window)
    
    freq_res = fs / len(signal)
    
    print(f"freq_res: {freq_res}")

    sum_sq_numerator = 0.0   # sideband sum (n != 0 )
    sum_sq_denominator = 0.0 # sum of everything (n = -max to +max)
    print(f"carrier: {f_carrier} Hz, modulator: {f_mod} Hz)")
    for n in range(-n_max, n_max + 1):
        # target frequency is f2 + n*f1
        target_freq = f_carrier + (n * f_mod)
        
        # #don't check negative frequencies or above nyquist
        if target_freq <= 0 or target_freq >= fs / 2:
            continue
            
        # find target index for the fft bins
        target_idx = int(round(target_freq / freq_res))
        
        # search range
        start = max(0, target_idx - search_window)
        end = min(len(magnitudes), target_idx + search_window + 1)
        
        # find peaks in the range
        peak_mag = np.max(magnitudes[start:end])
        
        # square it
        sq_val = peak_mag ** 2
        
        # and add to denominator sum
        sum_sq_denominator += sq_val
        
        # add to distortion sum, but only if it is not n != 0 
        if n != 0:
            sum_sq_numerator += sq_val
            # print sidebands?
            if peak_mag > 0.001: # filter out very low signals
                print(f"n={n}: peak at {target_idx*freq_res:.1f} Hz with amplitude {peak_mag:.4f}")
        else:
             print(f"carrer at n=0: peak at {target_idx*freq_res:.1f}Hz with amplitude {peak_mag:.4f}")

    # numerator: sum of sidfebands
    numerator = np.sqrt(sum_sq_numerator)
    
    # denominator: total sum
    denominator = np.sqrt(sum_sq_denominator)
    
    if denominator == 0:
        return 0.0
        
    timd = numerator / denominator
    return timd

def generate_timd_signal(f_carrier=10000, f_mod=500):
    fs = 48000 # 2*96000
    duration = 1
    t = np.linspace(0, duration, int(fs * duration))
    print(len(t))

    carrier = 0.8 * np.sin(2 * np.pi * f_carrier * t)

    sb_lower = 0.1 * np.sin(2 * np.pi * (f_carrier - f_mod) * t)
    sb_upper = 0.1 * np.sin(2 * np.pi * (f_carrier + f_mod) * t)
    
    sb_lower_2 = 0.05 * np.sin(2 * np.pi * (f_carrier - 2 * f_mod) * t)
    sb_upper_2 = 0.05 * np.sin(2 * np.pi * (f_carrier + 2 * f_mod) * t)
    
    # sum carrier and sidebands
    signal = carrier + sb_lower + sb_upper + sb_lower_2 + sb_upper_2
    
    return fs, signal

def generate_pink_noise(N, fs, fmin=1.0, fmax=None):
    if fmax is None:
        fmax = fs / 2

    X = np.fft.rfft(np.random.randn(N))
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # avoid DC + apply band limits
    freqs[0] = fmin
    mask = (freqs >= fmin) & (freqs <= fmax)

    X[mask] /= np.sqrt(freqs[mask])
    X[~mask] = 0.0

    x = np.fft.irfft(X, n=N)
    return x / np.std(x)

def calculate_min_fs(F):
    eigenvalues, _ = np.linalg.eig(F)

    min_required_fs = 0.0

    # lam = lambda, ie. go through all eigenvalues for F
    for lam in eigenvalues:
        real_part = np.real(lam)
        # min_fs >= -lambda^2/(2*real_part)
        mag_squared = np.abs(lam)**2
        req_fs = mag_squared / (-2 * real_part)
        
        # take max value for minimum fs
        if req_fs > min_required_fs:
            min_required_fs = req_fs
    
    return min_required_fs

def check_stability(F, fs):
    """
    check if (I + F/fs) = (Ihas eigenvalues all <= 1
    """
    dim = F.shape[0]
    I = np.eye(dim)
    T_s = 1/fs
    
    # Matrix: I + Ts*F
    discretized_matrix = I + (F * T_s)
    
    eigs, _ = np.linalg.eig(discretized_matrix)
    max_abs = np.max(np.abs(eigs))
    
    is_stable = max_abs <= 1.0
    print(f"checking fs={fs:.1f} Hz => max eigenvalue magnitude: {max_abs:.4f}")
    return is_stable

def init_latex():
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16
    })

def plot_mag(fs, resps, legends, title, xlim=[20,20e3], ylim=None, save=None, ylabel="Magnitude $\\left| H \\right|$ / dB re 1 V/V"):
    init_latex()
    plt.figure(figsize=(14, 5))
    for f, resp, legend in zip(fs, resps, legends):
        plt.semilogx(f, 20*np.log10(np.abs(resp)), label=legend)
    plt.xlabel("Frequency \\textit{f} / Hz")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()

def make_spectrum(x, fs, scaling=False, oneside=False):
    """
       freq, Y, YDB = engutil.make_spectrum(x, fs, scaling=False, oneside=False)
        Calculates the frequency spectrum of a signal with correct scaling.

        If 'oneside' and 'scaling' are both True, it computes a one-sided 
        amplitude spectrum. Otherwise, it computes a standard FFT.

        Args:
            x (array-like): Input signal array.
            fs (int or float): Sampling frequency.
            scaling (bool): If True, applies amplitude scaling.
            oneside (bool): If True, returns a one-sided spectrum.

        Returns:
            tuple: A tuple containing (freq, Y, YDB)
                - freq (np.ndarray): Frequency vector.
                - Y (np.ndarray): Complex FFT result (scaled if requested).
                - YDB (np.ndarray): FFT result in decibels.

       
    """
    x = np.asarray(x)
    N = len(x)

    if oneside:
        # Use rfft for efficiency with real signals, as it computes
        # only the positive frequency components.
        Y = np.fft.rfft(x)
        freq = np.fft.rfftfreq(N, d=1 / fs)

        if scaling:
         
            Y = Y / N
            
            Y[1:] *= 2

            if N % 2 == 0:
               
                Y[-1] /= 2
    else:
        # For a standard two-sided spectrum
        Y = np.fft.fft(x)
        freq = np.fft.fftfreq(N, d=1 / fs)
        if scaling:
            # For a two-sided spectrum, the standard amplitude scaling is 1/N.
            Y = Y / N

    # Calculate decibels for the final spectrum.
    # A small constant is added to avoid an error from log10(0).
    YDB = 20 * np.log10(np.abs(Y) + 1e-9)

    return freq, Y, YDB




