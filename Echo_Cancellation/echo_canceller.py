import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from scipy.io import wavfile

original_signal, sr = librosa.load("audio.wav", sr=16000)

print("Generating artificial echo...")
delay_samples = int(0.1 * sr)
h_room = np.zeros(delay_samples + 1)
h_room[0] = 1
alpha_echo = 0.7
h_room[-1] = alpha_echo

echo_signal = signal.convolve(original_signal, h_room)

print("Estimating alpha...")

corr = np.correlate(echo_signal, echo_signal, mode='full')
lag0_index = len(echo_signal) - 1
corr_pos = corr[lag0_index:]
alpha_est = corr_pos[delay_samples] / corr_pos[0]
r = alpha_est
alpha_corrected = (1 - np.sqrt(1 - 4*r*r)) / (2*r)

print(f"Estimated alpha: {alpha_corrected:.4f}")

print("Cleaning echo using inverse filtering...")
inverse_signal = np.zeros_like(echo_signal)

for n in range(len(echo_signal)):
    if n < delay_samples:
        inverse_signal[n] = echo_signal[n]
    else:
        inverse_signal[n] = echo_signal[n] - alpha_corrected * inverse_signal[n - delay_samples]

y_int16 = np.int16(echo_signal / np.max(np.abs(echo_signal)) * 32767)
wavfile.write("echo_signal.wav", sr, y_int16)

y_int16 = np.int16(inverse_signal / np.max(np.abs(inverse_signal)) * 32767)
wavfile.write("inverse_signal.wav", sr, y_int16)

print("Done")

