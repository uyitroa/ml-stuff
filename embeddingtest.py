import tensorflow as tf
import numpy as np

docs = [
	'Never coming back!',
	'horrible service',
	'rude waitress',
	'cold food',
	'horrible food!',
	'awesome',
	'awesome services!',
	'rocks',
	'poor work',
	'couldn\'t have done better'
]

labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0])
vocab_size = 50
encoded_docs = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

print(padded_reviews)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input((1,), dtype="string"))
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, output_mode="int",
                                                    output_sequence_length=max_length)
vectorize_layer.adapt(docs)
model.add(vectorize_layer)
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length)
model.add(embedding_layer)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

model.fit(np.array(docs), labels, epochs=100)

# print(embedding_layer.get_weights()[0])

print(model.predict(["horrible", "better"]))
