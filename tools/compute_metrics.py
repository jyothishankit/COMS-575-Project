import numpy as np
from scipy.io import wavfile
import math
from scipy import signal

def compute_metrics(cleanFile, enhancedFile, Fs, path):
    if path == 1:
        sampling_rate1, data1 = wavfile.read(cleanFile)
        sampling_rate2, data2 = wavfile.read(enhancedFile)
        if sampling_rate1 != sampling_rate2:
            raise ValueError("The two files do not match!\n")
    else:
        data1 = cleanFile
        data2 = enhancedFile
        sampling_rate1 = Fs
        sampling_rate2 = Fs
    if len(data1) != len(data2):
        length = min(len(data1), len(data2))
        data1 = data1[0:length] + np.spacing(1)
        data2 = data2[0:length] + np.spacing(1)
    # compute the SSNR and STOI
    snr_dist, segsnr_dist = snr(data1, data2, sampling_rate1)
    segSNR = np.mean(segsnr_dist)
    STOI = stoi(data1, data2, sampling_rate1)
    return segSNR, STOI

def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Both Speech Files must be same length.")

    overall_snr = 10 * np.log10(
        np.sum(np.square(clean_speech))
        / np.sum(np.square(clean_speech - processed_speech))
    )

    # Global Variables
    winlength = round(30 * sample_rate / 1000)  # window length in samples
    skiprate = math.floor(winlength / 4)  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )  # number of frames
    start = 0  # starting sample
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(
            signal_energy / (noise_energy + EPS) + EPS
        )
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)
        start = start + skiprate

    return overall_snr, segmental_snr


def stoi(x, y, fs_signal):
    if np.size(x) != np.size(y):
        raise ValueError("x and y should have the same length")

    # initialization, pay attention to the range of x and y(divide by 32768?)
    fs = 10000  # sample rate of proposed intelligibility measure
    N_frame = 256  # window support
    K = 512  # FFT size
    J = 15  # Number of 1/3 octave bands
    mn = 150  # Center frequency of first 1/3 octave band in Hz
    H, _ = thirdoct(fs, K, J, mn)  # Get 1/3 octave band matrix
    N = 30  # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15  # lower SDR-bound
    dyn_range = 40  # speech dynamic range

    # resample signals if other sample rate is used than fs
    if fs_signal != fs:
        x = signal.resample_poly(x, fs, fs_signal)
        y = signal.resample_poly(y, fs, fs_signal)

    # remove silent frames
    x, y = remove_silent_frames(x, y, dyn_range, N_frame, int(N_frame / 2))

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x, N_frame, N_frame / 2, K)  # apply short-time DFT to clean speech
    y_hat = stdft(
        y, N_frame, N_frame / 2, K
    )  # apply short-time DFT to processed speech

    x_hat = np.transpose(
        x_hat[:, 0 : (int(K / 2) + 1)]
    )  # take clean single-sided spectrum
    y_hat = np.transpose(
        y_hat[:, 0 : (int(K / 2) + 1)]
    )  # take processed single-sided spectrum

    X = np.sqrt(
        np.matmul(H, np.square(np.abs(x_hat)))
    )  # apply 1/3 octave bands as described in Eq.(1) [1]
    Y = np.sqrt(np.matmul(H, np.square(np.abs(y_hat))))

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm = np.zeros(np.size(np.arange(N - 1, x_hat.shape[1])))
    # init memory for intermediate intelligibility measure
    c = 10 ** (-Beta / 20)
    # constant for clipping procedure

    for m in range(N - 1, x_hat.shape[1]):
        X_seg = X[
            :, (m - N + 1) : (m + 1)
        ]  # region with length N of clean TF-units for all j
        Y_seg = Y[
            :, (m - N + 1) : (m + 1)
        ]  # region with length N of processed TF-units for all j
        # obtain scale factor for normalizing processed TF-region for all j
        alpha = np.sqrt(
            np.divide(
                np.sum(np.square(X_seg), axis=1, keepdims=True),
                np.sum(np.square(Y_seg), axis=1, keepdims=True),
            )
        )
        # obtain \alpha*Y_j(n) from Eq.(2) [1]
        aY_seg = np.multiply(Y_seg, alpha)
        # apply clipping from Eq.(3)
        Y_prime = np.minimum(aY_seg, X_seg + X_seg * c)
        # obtain correlation coeffecient from Eq.(4) [1]
        d_interm[m - N + 1] = column_vector_correlation(X_seg, Y_prime) / J

    d = (
        d_interm.mean()
    )  # combine all intermediate intelligibility measures as in Eq.(4) [1]
    return d


def thirdoct(fs, N_fft, numBands, mn):
    """
    [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
    inputs:
        FS:         samplerate
        N_FFT:      FFT size
        NUMBANDS:   number of bands
        MN:         center frequency of first 1/3 octave band
    outputs:
        A:          octave band matrix
        CF:         center frequencies
    """
    f = np.linspace(0, fs, N_fft + 1)
    f = f[0 : int(N_fft / 2 + 1)]
    k = np.arange(numBands)
    cf = np.multiply(np.power(2, k / 3), mn)
    fl = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k - 1) / 3), mn),
        )
    )
    fr = np.sqrt(
        np.multiply(
            np.multiply(np.power(2, k / 3), mn),
            np.multiply(np.power(2, (k + 1) / 3), mn),
        )
    )
    A = np.zeros((numBands, len(f)))

    for i in range(np.size(cf)):
        b = np.argmin((f - fl[i]) ** 2)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f - fr[i]) ** 2)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii:fr_ii] = 1

    rnk = np.sum(A, axis=1)
    end = np.size(rnk)
    rnk_back = rnk[1:end]
    rnk_before = rnk[0 : (end - 1)]
    for i in range(np.size(rnk_back)):
        if (rnk_back[i] >= rnk_before[i]) and (rnk_back[i] != 0):
            result = i
    numBands = result + 2
    A = A[0:numBands, :]
    cf = cf[0:numBands]
    return A, cf


def stdft(x, N, K, N_fft):
    """
    X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time hanning-windowed dft of X with frame-size N,
    overlap K and DFT size N_FFT. The columns and rows of X_STDFT denote the frame-index and dft-bin index,
    respectively.
    """
    frames_size = int((np.size(x) - N) / K)
    w = signal.windows.hann(N + 2)
    w = w[1 : N + 1]

    x_stdft = signal.stft(
        x,
        window=w,
        nperseg=N,
        noverlap=K,
        nfft=N_fft,
        return_onesided=False,
        boundary=None,
    )[2]
    x_stdft = np.transpose(x_stdft)[0:frames_size, :]

    return x_stdft


def remove_silent_frames(signal_x, signal_y, energy_range, frame_length, overlap):
    """
    Removes silent frames from signals x and y based on their energy levels.

    Parameters:
    signal_x (numpy.ndarray): Input signal x.
    signal_y (numpy.ndarray): Input signal y.
    energy_range (float): Energy range threshold for frame removal.
    frame_length (int): Length of the frame.
    overlap (int): Overlap between frames.

    Returns:
    numpy.ndarray, numpy.ndarray: Reconstructed signals x and y excluding silent frames.
    """
    # Create frames
    frames_indices = np.arange(0, (np.size(signal_x) - frame_length), overlap)
    
    # Define Hann window
    window = signal.windows.hann(frame_length + 2)
    window = window[1 : frame_length + 1]

    # Create indices for frames
    frame_indices_list = np.empty((np.size(frames_indices), frame_length), dtype=int)
    for j in range(np.size(frames_indices)):
        frame_indices_list[j, :] = np.arange(frames_indices[j] - 1, frames_indices[j] + frame_length - 1)

    # Calculate energy of each frame
    energy = 20 * np.log10(np.divide(np.linalg.norm(np.multiply(signal_x[frame_indices_list], window), axis=1), np.sqrt(frame_length)))
    
    # Determine silent frames based on energy range
    silent_frames_mask = (energy - np.max(energy) + energy_range) > 0
    count = 0

    # Reconstruct signals excluding silent frames
    reconstructed_signal_x = np.zeros(np.size(signal_x))
    reconstructed_signal_y = np.zeros(np.size(signal_y))
    for j in range(np.size(frames_indices)):
        if silent_frames_mask[j]:
            frame_indices_inner = np.arange(frames_indices[j], frames_indices[j] + frame_length)
            frame_indices_outer = np.arange(frames_indices[count], frames_indices[count] + frame_length)
            reconstructed_signal_x[frame_indices_outer] += np.multiply(signal_x[frame_indices_inner], window)
            reconstructed_signal_y[frame_indices_outer] += np.multiply(signal_y[frame_indices_inner], window)
            count += 1

    # Trim reconstructed signals
    reconstructed_signal_x = reconstructed_signal_x[0 : frame_indices_outer[-1] + 1]
    reconstructed_signal_y = reconstructed_signal_y[0 : frame_indices_outer[-1] + 1]
    return reconstructed_signal_x, reconstructed_signal_y

def column_vector_correlation(x_column_vector, y_column_vector):
    """
    Calculates the correlation coefficient between two column vectors x_column_vector and y_column_vector.
    Equivalent to 'corr' function from the statistics toolbox.
    
    Parameters:
    x_column_vector (numpy.ndarray): Input column vector x.
    y_column_vector (numpy.ndarray): Input column vector y.
    
    Returns:
    float: The correlation coefficient between x_column_vector and y_column_vector.
    """
    # Normalize x_column_vector
    x_normalized = np.subtract(x_column_vector, np.mean(x_column_vector, axis=1, keepdims=True))
    x_normalized = np.divide(x_normalized, np.linalg.norm(x_normalized, axis=1, keepdims=True))
    
    # Normalize y_column_vector
    y_normalized = np.subtract(y_column_vector, np.mean(y_column_vector, axis=1, keepdims=True))
    y_normalized = np.divide(y_normalized, np.linalg.norm(y_normalized, axis=1, keepdims=True))
    
    # Calculate the correlation coefficient using matrix multiplication
    rho = np.trace(np.matmul(x_normalized, np.transpose(y_normalized)))
    
    return rho
