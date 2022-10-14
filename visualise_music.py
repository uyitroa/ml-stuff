import numpy as np
from scipy import signal
import pydub
import cv2
import os
from scipy.io.wavfile import write

freq_size = 4
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.2
fontColor = (255, 255, 255)
thickness = 1
lineType = 2


def read(f, normalized=False):
	"""MP3 to numpy array"""
	a = pydub.AudioSegment.from_mp3(f)
	y = np.array(a.get_array_of_samples())
	if a.channels == 2:
		y = y.reshape((-1, 2))
	if normalized:
		return a.frame_rate, np.float32(y) / 2 ** 15
	else:
		return a.frame_rate, y


def get_specgram(x, sample_rate):
	if len(x.shape) >= 2:
		x = x[:, 0]
	x = x[x.shape[0] // 2:]
	freqs, t, spec = signal.spectrogram(x, fs=sample_rate, nperseg=2048*2, noverlap=2048)
	# spec = 20.0 * np.log10(spec + 10e-3)
	spec = spec ** 0.4 * 2000
	return spec, freqs, t, x


def find_closest(A, target):
	idx = A.searchsorted(target)
	idx = np.clip(idx, 1, len(A) - 1)
	left = A[idx - 1]
	right = A[idx]
	idx -= target - left < right - target
	return idx


music = "sao.mp3"
sample_rate, x = read(music, normalized=True)

spec, freqs, t, x = get_specgram(x, sample_rate)

filter_freq_index = find_closest(freqs, 10000)
freqs = freqs[:filter_freq_index]
spec = spec[:filter_freq_index, :]

audio_length = x.shape[0] / sample_rate
height = 500
n_freq_toshow = 200
freq_size = int(height / n_freq_toshow)
fps = 60.0

writer = cv2.VideoWriter("test.mkv", cv2.VideoWriter_fourcc(*"X264"), fps, (200, height))
audio_fps = t.shape[0] / audio_length

maxfreq = np.max(freqs)
n_frame = int(audio_length * fps)

# for i in range(freqs.shape[0]):
# 	print(freqs[i])
start = 1
end = 1.5

a = np.linspace(start, end, n_freq_toshow)
a = (np.exp(a*a) - np.exp(start * start)) * (freqs.shape[0] - 1) / (np.exp(end*end) - np.exp(start * start))

for i in a:
	print(i)

print(freqs.shape[0] - 1)

# for i in t:
# 	print(i)
#
# for i in freqs:
# 	print("{} hz".format(i))

for i in range(n_frame):
	np_img = np.zeros((height, 200, 3), dtype=np.uint8)

	t_index = find_closest(t, i / n_frame * audio_length)
	for (index, j) in enumerate(a):
		y = min(round(j), freqs.shape[0]-1)
		np_img[index * freq_size:index * freq_size + (freq_size - 1), :int(max(0, spec[y, t_index])), :] = 255
	writer.write(np_img)

writer.release()
write("audio.mp3", round(sample_rate), x)
os.system('ffmpeg -i test.mkv -codec copy test.mp4 -y')
os.system("ffmpeg -i test.mp4 -i audio.mp3 -c:v copy -c:a aac what.mp4 -y")
