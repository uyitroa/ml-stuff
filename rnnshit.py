import tensorflow as tf
import numpy as np

train_data = tf.keras.utils.text_dataset_from_directory("movie-reviews-dataset/train")
test_data = tf.keras.utils.text_dataset_from_directory("movie-reviews-dataset/test")


def get_rnn(train_texts_to_adapt, vocab_size=1000, sentence_length=100):
	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Input((1,), dtype="string"))
	vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int",
	                                                    output_sequence_length=sentence_length)
	vectorize_layer.adapt(train_texts_to_adapt)
	model.add(vectorize_layer)
	model.add(tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128))
	model.add(tf.keras.layers.LSTM(units=64))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(units=64, activation='relu'))
	model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
	return model


train_text = train_data.map(lambda text, labels: text)
model = get_rnn(train_text)

model.fit(train_data, epochs=1, validation_data=test_data)

print(model.predict(["I love this", "this is bad", "holy shit this is good", "I love this shit"]))