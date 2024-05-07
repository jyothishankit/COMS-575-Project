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
    snr_dist, segsnr_dist = snr(data1, data2, sampling_rate1)
    segSNR = np.mean(segsnr_dist)
    STOI = stoi(data1, data2, sampling_rate1)
    return segSNR, STOI

def snr(clean_speech, processed_speech, sample_rate):
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError("Same size required for clean and generated")

    overall_snr = 10 * np.log10(
        np.sum(np.square(clean_speech))
        / np.sum(np.square(clean_speech - processed_speech))
    )

    winlength = round(30 * sample_rate / 1000)
    skiprate = math.floor(winlength / 4)
    MIN_SNR = -10
    MAX_SNR = 35
    num_frames = int(
        clean_length / skiprate - (winlength / skiprate)
    )
    start = 0
    window = 0.5 * (
        1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1))
    )

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        clean_frame = clean_speech[start : start + winlength]
        processed_frame = processed_speech[start : start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)
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
        raise ValueError("Same size required for x and y")
    fs = 10000
    N_frame = 256
    K = 512
    J = 15
    mn = 150
    H, _ = thirdoct(fs, K, J, mn)
    N = 30
    Beta = -15
    dyn_range = 40
    if fs_signal != fs:
        x = signal.resample_poly(x, fs, fs_signal)
        y = signal.resample_poly(y, fs, fs_signal)
    x, y = remove_silent_frames(x, y, dyn_range, N_frame, int(N_frame / 2))
    x_hat = stdft(x, N_frame, N_frame / 2, K)
    y_hat = stdft(
        y, N_frame, N_frame / 2, K
    )
    x_hat = np.transpose(
        x_hat[:, 0 : (int(K / 2) + 1)]
    )
    y_hat = np.transpose(
        y_hat[:, 0 : (int(K / 2) + 1)]
    )
    X = np.sqrt(
        np.matmul(H, np.square(np.abs(x_hat)))
    )
    Y = np.sqrt(np.matmul(H, np.square(np.abs(y_hat))))
    d_interm = np.zeros(np.size(np.arange(N - 1, x_hat.shape[1])))
    c = 10 ** (-Beta / 20)
    for m in range(N - 1, x_hat.shape[1]):
        X_seg = X[
            :, (m - N + 1) : (m + 1)
        ]
        Y_seg = Y[
            :, (m - N + 1) : (m + 1)
        ]
        alpha = np.sqrt(
            np.divide(
                np.sum(np.square(X_seg), axis=1, keepdims=True),
                np.sum(np.square(Y_seg), axis=1, keepdims=True),
            )
        )
        aY_seg = np.multiply(Y_seg, alpha)
        Y_prime = np.minimum(aY_seg, X_seg + X_seg * c)
        d_interm[m - N + 1] = column_vector_correlation(X_seg, Y_prime) / J

    d = (
        d_interm.mean()
    )
    return d


def thirdoct(fs, N_fft, numBands, mn):
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
    frames_indices = np.arange(0, (np.size(signal_x) - frame_length), overlap)
    window = signal.windows.hann(frame_length + 2)
    window = window[1 : frame_length + 1]
    frame_indices_list = np.empty((np.size(frames_indices), frame_length), dtype=int)
    for j in range(np.size(frames_indices)):
        frame_indices_list[j, :] = np.arange(frames_indices[j] - 1, frames_indices[j] + frame_length - 1)
    energy = 20 * np.log10(np.divide(np.linalg.norm(np.multiply(signal_x[frame_indices_list], window), axis=1), np.sqrt(frame_length)))
    silent_frames_mask = (energy - np.max(energy) + energy_range) > 0
    count = 0
    reconstructed_signal_x = np.zeros(np.size(signal_x))
    reconstructed_signal_y = np.zeros(np.size(signal_y))
    for j in range(np.size(frames_indices)):
        if silent_frames_mask[j]:
            frame_indices_inner = np.arange(frames_indices[j], frames_indices[j] + frame_length)
            frame_indices_outer = np.arange(frames_indices[count], frames_indices[count] + frame_length)
            reconstructed_signal_x[frame_indices_outer] += np.multiply(signal_x[frame_indices_inner], window)
            reconstructed_signal_y[frame_indices_outer] += np.multiply(signal_y[frame_indices_inner], window)
            count += 1
    reconstructed_signal_x = reconstructed_signal_x[0 : frame_indices_outer[-1] + 1]
    reconstructed_signal_y = reconstructed_signal_y[0 : frame_indices_outer[-1] + 1]
    return reconstructed_signal_x, reconstructed_signal_y

def column_vector_correlation(x_column_vector, y_column_vector):
    x_normalized = np.subtract(x_column_vector, np.mean(x_column_vector, axis=1, keepdims=True))
    x_normalized = np.divide(x_normalized, np.linalg.norm(x_normalized, axis=1, keepdims=True))
    y_normalized = np.subtract(y_column_vector, np.mean(y_column_vector, axis=1, keepdims=True))
    y_normalized = np.divide(y_normalized, np.linalg.norm(y_normalized, axis=1, keepdims=True))
    rho = np.trace(np.matmul(x_normalized, np.transpose(y_normalized)))
    return rho
