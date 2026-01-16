import numpy as np
import scipy as sp
from scipy.signal.windows import flattop
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np 
import engutil
import solver
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp


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

def save_wav(path_n_name, signal, fs):
    signal_int16 = np.int16(signal / np.max(np.abs(signal)) * 32767)
    wavfile.write(path_n_name, fs, signal_int16)   

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
    
    #print(f"freq_res: {freq_res}")

    sum_sq_numerator = 0.0   # sideband sum (n != 0 )
    sum_sq_denominator = 0.0 # sum of everything (n = -max to +max)
    #print(f"carrier: {f_carrier} Hz, modulator: {f_mod} Hz)")
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

def welchie(u, X, fs):
    # print(data.keys())
    # fs = data['sample_rate'][0][0]
    i = X[0]
    d = X[1]
    v = X[2]

    # print(f"len: {len(u)}")
    # print(f"nperseg = {len(u)/25}")
    # print(f"len: {len(u)/fs}")
    numsamples = int(len(u))
    numsecs = len(u)/fs
    numavgs = 15 # is really 2*numavgs

    print(f"len samples: {len(u)}")
    print(f"nperseg = {len(u)/numavgs}")
    print(f"len seconds: {len(u)/fs}")
    #print(fs)
    # 96kHz 
    nperseg = int(len(u)/numavgs) # 96000 #2**16
    noverlap = nperseg//2 # 2**8 #// 2
    window = 'hann'
    #nfft = 2**17

    # print(f"Num avg: {2*numsamples/nperseg}")
    # print(f"freq res: {fs/nperseg}")
    
    f, S_uu = sp.signal.welch(u, fs, window, nperseg, noverlap) # , nfft=nfft)
    f, S_iu = sp.signal.csd(u, i, fs, window, nperseg, noverlap)
    f, S_du = sp.signal.csd(u, d, fs, window, nperseg, noverlap)
    f, S_vu = sp.signal.csd(u, v, fs, window, nperseg, noverlap)

    mu =  0# 1e-7# 1e-7
    
    G_iu = S_iu/(S_uu + mu)
    G_du = S_du/(S_uu + mu)
    G_vu = S_vu/(S_uu + mu)

    return G_iu, G_du, G_vu, f

def solve_forward_euler(F, G, u_signal, x0, fs):
    """
    Simulates a state-space system using Forward Euler.
    
    Parameters:
    F, G: System matrices
    u_signal: Array of inputs over time
    x0: Initial state vector
    fs: Sampling frequency
    """
    Ts = 1 / fs
    num_steps = len(u_signal)
    
    # 1. Initialize History Arrays
    # We need to store the state at every time step to plot it later.
    # Shape: (Number of Time Steps, Number of States)
    num_states = len(x0)
    x_history = np.zeros((num_steps, num_states))
    
    # Set current state to initial state
    x_curr = x0.copy()
    
    #print(f"Simulating {num_steps} steps with Ts={Ts:.10f}s...")

    # 2. The Simulation Loop
    for n in range(num_steps):
        # Store current state
        x_history[n] = x_curr
        
        # Get current input (handle scalar or vector inputs)
        u_curr = u_signal[n]
        
        # --- THE FORMULA FROM YOUR IMAGE ---
        # Calculate the derivative (slope)
        # dx/dt = F*x + G*u
        dx = (F @ x_curr) + (G * u_curr)
        
        # Euler Step: New = Old + (Slope * StepSize)
        x_next = x_curr + (dx * Ts)
        # -----------------------------------
        
        # Update for next iteration
        x_curr = x_next
        
    return x_history

def solve_forward_euler_optimized(F, G, u_signal, x0, fs):
    Ts = 1/fs
    num_states = len(x0)
    I = np.eye(num_states)
    
    # Pre-compute Discrete Matrices
    Phi = I + (F * Ts)
    Gamma = G * Ts
    
    # Initialize
    x_history = np.zeros((len(u_signal), num_states))
    x_curr = x0.copy()
    
    # Faster Loop
    for n in range(len(u_signal)):
        x_history[n] = x_curr
        
        # Single matrix multiply + add
        x_next = (Phi @ x_curr) + (Gamma * u_signal[n])
        
        x_curr = x_next
        
    return x_history

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

def midpoint_forward_euler(F, G, u_signal, x0, fs):
    Ts = 1 / fs
    num_steps = len(u_signal)
    
    # 1. Initialize History Arrays
    # We need to store the state at every time step to plot it later.
    # Shape: (Number of Time Steps, Number of States)
    num_states = len(x0)
    x_history = np.zeros((num_steps, num_states))
    
    # Set current state to initial state
    x_curr = x0.copy()

    for n in range(num_steps):
        x_history[n] = x_curr
        u_curr = u_signal[n]

        # Calculate slope at start point to find midpoint
        dx1 = (F @ x_curr) + (G * u_curr)
        x_mid = x_curr + 0.5 * Ts * dx1

        # Signal at midpoint
        # Right now it uses signal at starting point. Not the best solution u_mid = 0.5 * (u_signal[n] + u_signal[n+1]) solution imporves accuracy but i get out of bounds for the last sample...
        
        # u_mid = 0.5 * (u_signal[n] + u_signal[n+1])
        u_mid = u_curr
        
        # Slope at midpoint
        dx2 = (F @ x_mid) + (G * u_mid)

        x_next = x_curr + Ts * dx2

        x_curr = x_next

    return x_history

def vel_2_spl(vel, r, f, log=False):
    # In:
    # Vel: Velocity signal
    # r: radius of driver
    # Out: Prms at 1m in dB SPL
    rho = 1.21
    c = 344
    k = 2*np.pi*f/c
    v_rms = vel/np.sqrt(2)
    Sd = r**2 * np.pi
    U = v_rms * Sd
    p_rms = rho*c*k*U/(4*np.pi*1)
    p_rms_dB = 20*np.log10(p_rms/(20e-6))
    if log:
        return p_rms_dB
    else:
        return p_rms

def plot_spectrum_in_spl(x, fs, radius, xlim=None, ylim=None, window=False, log=False, title="Spectrum in SPL RMS", ylabel="Magnitude / Pa", xlabel="Frequency / Hz", save=None):
    """
    Plot the one-sided magnitude spectrum with frequency in Hz.

    Parameters
    ----------
    x : array_like
        Time-domain signal
    fs : float
        Sampling frequency in Hz
    """

    x = np.asarray(x)
    N = len(x)

    # One-sided FFT
    X = np.fft.rfft(x)

    # Frequency axis in Hz (NOT sample index)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Amplitude normalization
    mag = np.abs(X) / N
    mag[1:-1] *= 2  # keep DC and Nyquist correct

    #Convert from vel to spl. Also convert from peak to rms
    for n in range(len(mag)):
        mag[n] = vel_2_spl(mag[n], radius, freqs[n], log=log)

    # Plot
    init_latex()
    plt.figure(figsize=(12,6))
    plt.semilogx(freqs, mag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)    
    plt.grid(True)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()

def plot_spectrum(x, fs, xlim=None, ylim=None, window=False, title="Spectrum", ylabel="Magnitude", xlabel="Frequency / Hz", save=None):
    """
    Plot the one-sided magnitude spectrum with frequency in Hz.

    Parameters
    ----------
    x : array_like
        Time-domain signal
    fs : float
        Sampling frequency in Hz
    """

    x = np.asarray(x)
    N = len(x)

    # One-sided FFT
    X = np.fft.rfft(x)

    # Frequency axis in Hz (NOT sample index)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Amplitude normalization
    mag = np.abs(X) / N
    mag[1:-1] *= 2  # keep DC and Nyquist correct

    # Convert to RMS
    mag = mag/np.sqrt(2)

    # Plot
    init_latex()
    plt.figure(figsize=(12,6))
    plt.semilogx(freqs, mag)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)    
    plt.grid(True)
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    plt.show()

def thd_spl(signal, fs, radius, max_harmonic=19):
    # 1. apply hanning window - this attenuated by 0.5 which we will fix later
    window = np.hanning(len(signal))
    signal_windowed = signal * window
    
    # 2. perform FFT on the windowed signal
    yf = np.fft.rfft(signal_windowed)
    
    # 3. calculate magnitudes
    # we normalize by the window instead of the usual N and multiply by 2 for one-sided spectrum
    magnitudes = np.abs(yf) * 2 / np.sum(window)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)

    # 3.5 convert the magnitudes to SPL
    for n in range(len(magnitudes)):
        magnitudes[n] = vel_2_spl(magnitudes[n], radius, freqs[n], log=False)
    
    # 4. find the fundamental in the signal from the magnitudes, ie. not the 1st index (that would be DC) but the 2nd
    fundamental_idx = np.argmax(magnitudes[1:]) + 1
    fundamental_mag_peak = magnitudes[fundamental_idx]
    
    # 5. calculate frequency resolution
    freq_res = fs / len(signal)
    f1_freq = fundamental_idx * freq_res
    
    #print(f"Detected Fundamental: {f1_freq:.2f} Hz")

    # calculate THD sums
    sum_squares_harmonics = 0
    sum_squares_total = fundamental_mag_peak ** 2
    
    # store the harmonic data
    harmonic_data = {} 
    
    
    harmonic_data[1] = fundamental_mag_peak

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
            
            
            harmonic_data[n] = harm_mag_peak

    # compute THD_R 
    numerator = np.sqrt(sum_squares_harmonics)
    denominator = np.sqrt(sum_squares_total)
    
    # check for divide by zero 
    if denominator == 0:
        thd_r = 0
    else:
        thd_r = numerator / denominator

    return thd_r, harmonic_data

def timd_spl(signal, fs, f_mod, f_carrier, radius, n_max=3, search_window=5):
    # window = np.hanning(len(signal)) # Maybe use windowing 
    # window = np.blackman(len(signal))
    # window = np.kaiser(len(signal), 0)
    window = flattop(len(signal))
    signal_windowed = signal * window
    
    yf = np.fft.rfft(signal_windowed)
    # yf = np.fft.rfft(signal)
    magnitudes = np.abs(yf) * 2 / np.sum(window) # len(signal) # np.sum(window)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)

    # 3.5 convert the magnitudes to SPL
    for n in range(len(magnitudes)):
        magnitudes[n] = vel_2_spl(magnitudes[n], radius, freqs[n], log=False)
    
    freq_res = fs / len(signal)
    
    #print(f"freq_res: {freq_res}")

    sum_sq_numerator = 0.0   # sideband sum (n != 0 )
    sum_sq_denominator = 0.0 # sum of everything (n = -max to +max)
    #print(f"carrier: {f_carrier} Hz, modulator: {f_mod} Hz)")
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
            #if peak_mag > 0.001: # filter out very low signals
                #print(f"n={n}: peak at {target_idx*freq_res:.1f} Hz with amplitude {peak_mag:.4f}")
        #else:
            #print(f"carrer at n=0: peak at {target_idx*freq_res:.1f}Hz with amplitude {peak_mag:.4f}")

    # numerator: sum of sidfebands
    numerator = np.sqrt(sum_sq_numerator)
    
    # denominator: total sum
    denominator = np.sqrt(sum_sq_denominator)
    
    if denominator == 0:
        return 0.0
        
    timd = numerator / denominator
    return timd

def find_fundemental_pRMS(v, fs, freq, radius):
    yf = np.fft.rfft(v)
    magnitudes = np.abs(yf) * 2 / len(v)
    fundemental_peak = magnitudes[int(freq / (fs/len(v)))]
    Prms_sim = vel_2_spl(fundemental_peak, radius, freq)
    return Prms_sim


