import tensorflow as tf
import numpy as np
from tqdm import tqdm


def generate_data():
	myfile = open("trainingdata.txt", "r")
	mylist = eval(myfile.read())
	stringlist = list(map(lambda l: " ".join(l), mylist))
	dataset = []
	for s in stringlist:
		wordlist = s.split(" ")
		worddatalist = [wordlist[0]]
		for w in wordlist[1:]:
			worddatalist.append(w)
			dataset.append(" ".join(worddatalist))
	print(dataset[:15])

	return dataset


def get_rnn(texts_to_adapt):
	model = tf.keras.Sequential()

	# model.add(tf.keras.layers.Input((1,), dtype="string"))
	#
	# vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=20, output_mode="int", output_sequence_length=20)
	# vectorize_layer.adapt(texts_to_adapt)
	# model.add(vectorize_layer)

	model.add(tf.keras.layers.Embedding(input_dim=15, output_dim=20, input_length=15))
	model.add(tf.keras.layers.LSTM(units=128))
	model.add(tf.keras.layers.Dropout(0.4))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(15, activation='softmax'))

	loss = tf.keras.losses.SparseCategoricalCrossentropy()
	opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
	model.compile(loss=loss, optimizer=opt, metrics=['acc'])
	return model

#
# def train(dataset, model: tf.keras.Model, vectorize_layer, epochs, n_batch):
# 	dataset = np.array_split(dataset, n_batch)
#
# 	for epoch in range(epochs):
# 		progress_bar = tqdm(dataset)
# 		acc = 0
# 		loss = 0
# 		common = 0
# 		for j, data_batch in enumerate(progress_bar):
# 			data_batch = vectorize_layer(data_batch)
# 			cut_index = j % 3 + 1
#
# 			X = data_batch[..., :cut_index]
# 			Y = data_batch[..., cut_index:cut_index + 1]
#
# 			# print(X.shape, Y.shape)
# 			# tf.print(Y)
#
# 			Y = tf.reshape(Y, (Y.shape[0],))
# 			counts = np.bincount(Y)
# 			# print(cut_index)
# 			# print(np.argmax(counts))
# 			common += np.argmax(counts)
#
# 			d = model.train_on_batch(X, Y)
# 			loss += d[1]
# 			acc += d[0]
# 		print(f"Accuracy: {loss / len(dataset[0])}, idk: {acc / len(dataset[0])}")
# 		print(f"Most common: {common/len(dataset[0])}")
#

dataset = generate_data()
model = get_rnn(dataset)

# model.summary()
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=15, output_mode="int",
                                                    standardize="lower", output_sequence_length=15)
vectorize_layer.adapt(dataset)

data_batch = vectorize_layer(dataset).numpy()


Y = []
for s in data_batch:
	last_index = -1
	for w in s:
		if int(w) == 0:
			break
		last_index += 1
	Y.append(int(s[last_index]))
	s[last_index] = 0

Y = np.array(Y)
X = data_batch
print(X)
print(Y)

# train(dataset, model, vectorize_layer, 30, 64)
# max_sequence_len = max([len(x) for x in X])
# input_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_sequence_len, padding='pre'))
model.fit(X, Y, epochs=3)

novel = tf.keras.Sequential([tf.keras.layers.Input((1,), dtype="string"), vectorize_layer, model])
vocab = np.array(vectorize_layer.get_vocabulary())


def predict(stringlist):
	y = novel.predict(stringlist)
	print(y)
	a = np.argmax(y[..., :len(vocab)-1])
	return vocab[a]
