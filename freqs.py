import librosa
import numpy as np

filename = "audio.wav"
y, sr = librosa.load(filename)
n_fft = 1024
hop_length = 512

spec = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length))
freqs = librosa.core.fft_frequencies(n_fft=n_fft)
times = librosa.core.frames_to_time(spec[0], sr=sr, n_fft=n_fft, hop_length=hop_length)

print('spectrogram size', spec.shape)

fft_bin = 14
time_idx = 1000

print('freq (Hz)', freqs[fft_bin])
print('time (s)', times[time_idx])
print('amplitude', spec[fft_bin, time_idx])

print(spec.shape)
print(np.sum(times))