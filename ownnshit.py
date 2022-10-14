import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Hello")


def train_step(x_train, y_train, model: tf.keras.Model):
	with tf.GradientTape() as tape:
		y_pred = model(x_train)
		y_true = tf.convert_to_tensor(y_train)
		lossvalue = model.compiled_loss(y_true, y_pred, regularization_losses=model.losses)

		y_pred = y_pred.numpy()
		y_pred = np.argmax(y_pred, axis=1)
		acc = np.sum(y_train == y_pred)/y_train.size

	grad = tape.gradient(lossvalue, model.trainable_variables)
	model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

	return lossvalue, acc


# def test_step(x_test, y_test, model: tf.keras.Model):
# 	y_pred = model(x_test)
# 	lossvalue = model.compiled_loss

def fit(model, x_train, y_train, epochs, validation_data, batch_size=64):
	dataset = np.array_split(x_train, batch_size)

	for epoch in range(epochs):
		progress_bar = tqdm(dataset)
		for _, image_batch in enumerate(progress_bar):
			lossvalue, acc = train_step(x_train, y_train, model)


class OwnLayerShit(tf.keras.layers.Layer):
	def __init__(self, units=64, n_layers=2):
		super().__init__()
		self.units = units
		self.n_layers = 2
		self.denses = [tf.keras.layers.Dense(units, activation='relu') for _ in range(n_layers)]

	# def build(self, input_shape):
	# 	self.w = []
	# 	self.b = []
	#
	# 	self.w.append(self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True))
	# 	self.b.append(self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True))
	# 	for i in range(self.n_layers - 1):
	# 		self.w.append(self.add_weight(shape=(self.units, self.units), initializer='random_normal', trainable=True))
	# 		self.b.append(self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True))

	def call(self, inputs):
		for i in range(self.n_layers):
			# inputs = tf.keras.activations.relu(tf.matmul(inputs, self.w[i]) + self.b[i])
			inputs = self.denses[i](inputs)
		return inputs


class OwnNeuralNetworkShit(tf.keras.Model):
	def __init__(self, neuron_number, image_shape, n_class, n_layers):
		super().__init__()

		self.n_class = n_class
		self.n_layers = n_layers

		input_shape = 1
		for i in image_shape:
			input_shape *= i

		# self.W = [tf.Variable(tf.random.normal((input_shape if i == 0 else neuron_number, neuron_number)), trainable=True) for i in range(n_layers)]
		# self.B = [tf.Variable(tf.random.normal((neuron_number,)), trainable=True) for i in range(n_layers)]
		#
		# self.OW = tf.Variable(tf.random.normal((neuron_number, n_class)), trainable=True)
		# self.Ob = tf.Variable(tf.random.normal((n_class,)), trainable=True)

		self.dense1 = OwnLayerShit(64, n_layers=3)

		self.last = tf.keras.layers.Dense(n_class, activation='softmax')

	# def calculate(self, image):
	# 	# image = tf.expand_dims(image, 0)
	# 	image = tf.reshape(image, (1, image.shape[0]))
	#
	# 	for i in range(self.n_layers):
	# 		image = tf.keras.activations.relu(image @ self.W[i] + self.B[i])
	# 	out = tf.keras.activations.softmax(image @ self.OW + self.Ob)
	# 	return out

	def call(self, seq: tf.Tensor):
		# probs = tf.vectorized_map(lambda image: self.calculate(image), seq)

		# return tf.reshape(probs, (probs.shape[0], self.n_class))
		x = self.dense1(seq)
		y = self.last(x)

		return y

# def train_step(self, data):
# 	x, y = data
#
# 	with tf.GradientTape() as tape:
# 		y_pred = self(x, training=True)
# 		loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
#
# 	trainable_vars = self.trainable_variables
# 	gradients = tape.gradient(loss, trainable_vars)
# 	self.optimizer.apply_gradients(zip(gradients, trainable_vars))
# 	self.compiled_metrics.update_state(y, y_pred)
#
# 	return {m.name: m.result() for m in self.metrics}
#
# def test_step(self, data):
# 	x, y = data
#
# 	y_pred = self(x, training=False)
# 	self.compiled_loss(y, y_pred, regularization_losses=self.losses)
# 	self.compiled_metrics.update_state(y, y_pred)
# 	return {m.name: m.result() for m in self.metrics}


def get_onehot(a):
	a = a.flatten()
	C = np.zeros((a.size, 10))
	C[np.arange(a.size), a] = 1

	return C


def flatten_images(images):
	return images.reshape((images.shape[0], images[0].size))


x_train = flatten_images(x_train / 255)
x_test = flatten_images(x_test / 255)

# y_train = get_onehot(y_train)
# y_test = get_onehot(y_test)

train_data = ([x_train], [y_train])
valid_data = ([x_test], [y_test])

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']
loss = tf.keras.losses.SparseCategoricalCrossentropy()

print("OK")

model = tf.keras.Sequential([OwnNeuralNetworkShit(64, image_shape=x_train[0].shape, n_class=10, n_layers=2)])

print("Compiling")
model.compile(optimizer=opt, metrics=metrics, loss=loss)
print("Fitting")

fit(model, x_train, y_train, epochs=1, validation_data=(x_test, y_test))

print(model.summary())


def visualise_trainablevar(trainable_variables, input_shape):
	for variable in trainable_variables:
		name = variable.name
		data = variable.numpy()

		if len(data.shape) != 2:
			continue

		try:
			data = data.reshape((*input_shape, data.shape[-1]))
		except:
			continue
		print("Saving")
		for i in range(data.shape[-1]):
			node = data[:, :, i]
			image = np.zeros((*input_shape, 3))
			image.fill(100)

			image[:, :, 1] += (node > 0) * node * 300
			image[:, :, 2] -= (node < 0) * node * 300

			cv2.imwrite(f"hallo-{i}.png", image)


# visualise_trainablevar(model.trainable_variables, (28, 28))

y_pred = model.predict(x_test)

matr = tf.math.confusion_matrix(y_test, np.argmax(y_pred, axis=1), num_classes=10)

import matplotlib.pyplot as plt

plt.imshow(matr)
plt.colorbar()
plt.show()
