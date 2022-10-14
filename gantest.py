import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
y_train = y_train / 255

noise_size = 100

x_train = np.concatenate((x_train, x_test))


class Conv2DTReLU(tf.keras.layers.Layer):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.convt = tf.keras.layers.Conv2DTranspose(*args, **kwargs)
		self.normalize = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.LeakyReLU()

	def call(self, inputs, *args, **kwargs):
		inputs = self.convt(inputs)
		# inputs = self.normalize(inputs)
		inputs = self.relu(inputs)
		return inputs


class Conv2DReLUDrop(tf.keras.layers.Layer):
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.convt = tf.keras.layers.Conv2D(*args, **kwargs)
		self.normalize = tf.keras.layers.BatchNormalization()
		self.relu = tf.keras.layers.LeakyReLU()
		self.dropout = tf.keras.layers.Dropout(0.3)

	def call(self, inputs, *args, **kwargs):
		inputs = self.convt(inputs)
		# inputs = self.normalize(inputs)
		inputs = self.relu(inputs)
		inputs = self.dropout(inputs)
		return inputs


class Discriminator(tf.keras.Model):
	def __init__(self, input_shape):
		super().__init__()
		self.reshape = tf.keras.layers.Reshape(target_shape=(*input_shape, 1))
		self.conv1 = Conv2DReLUDrop(filters=64, kernel_size=3, input_shape=input_shape, padding='same')
		self.conv2 = Conv2DReLUDrop(filters=64, kernel_size=3, padding='same')
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
		self.dense_last = tf.keras.layers.Dense(units=1, activation='sigmoid')
		self.last = tf.keras.layers.Flatten()

	def call(self, inputs):
		inputs = self.reshape(inputs)
		inputs = self.conv1(inputs)
		inputs = self.conv2(inputs)
		inputs = self.flatten(inputs)
		inputs = self.dense1(inputs)
		inputs = self.dense_last(inputs)
		inputs = self.last(inputs)

		return inputs


class Generator(tf.keras.Model):
	def __init__(self, noise_size):
		super().__init__()

		units = 7 * 7 * 128
		shape = (7, 7, 128)

		self.dense1 = tf.keras.layers.Dense(units=units, activation='relu', input_shape=(noise_size,))
		self.reshape = tf.keras.layers.Reshape(target_shape=shape)
		self.convt1 = Conv2DTReLU(filters=64, kernel_size=3, strides=2, padding='same')  # -> (14, 14, ...)
		self.convt2 = Conv2DTReLU(filters=64, kernel_size=3, strides=2, padding='same')  # -> (28, 28, ...)
		self.conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), padding='same', activation='sigmoid')
		self.reshape2 = tf.keras.layers.Reshape(target_shape=(28, 28))

	def call(self, inputs):
		inputs = self.dense1(inputs)
		inputs = self.reshape(inputs)
		inputs = self.convt1(inputs)
		inputs = self.convt2(inputs)
		inputs = self.conv(inputs)
		inputs = self.reshape2(inputs)

		return inputs


class Gan(tf.keras.Model):
	def __init__(self, gen: tf.keras.Model, disc: tf.keras.Model):
		super().__init__()

		self.gen = gen

		disc.trainable = False  # trick, since disc is already compiled
		self.disc = disc

	def call(self, inputs):
		inputs = self.gen(inputs)
		inputs = self.disc(inputs)

		return inputs


def train_step(x_train, gen: tf.keras.Model, disc: tf.keras.Model, gan: tf.keras.Model):
	batch_size = x_train.shape[0]

	# with tf.GradientTape() as disc_tape:
	# 	noise_vector = tf.random.normal(shape=(batch_size, noise_size))
	# 	fake_numbers = gen(noise_vector)
	# 	fake_pred = disc(fake_numbers)
	# 	fake_results = tf.zeros((batch_size, 1))
	#
	# 	real_pred = disc(x_train)
	# 	real_results = tf.zeros((batch_size, 1))
	#
	# 	disc_real_loss = disc.compiled_loss(real_results, real_pred)
	# 	disc_fake_loss = disc.compiled_loss(fake_results, fake_pred)
	#
	# 	disc_loss = disc_real_loss + disc_fake_loss
	#
	# disc_grad = disc_tape.gradient(disc_loss, disc.trainable_variables)
	# disc.optimizer.apply_gradients(zip(disc_grad, disc.trainable_variables))
	#
	# with tf.GradientTape() as gen_tape:
	# 	noise_vector = tf.random.normal(shape=(batch_size, noise_size))
	# 	fake_numbers = gen(noise_vector)
	# 	fake_pred = disc(fake_numbers)
	#
	# 	real_result = tf.ones((batch_size, 1))
	#
	# 	gen_loss = gen.compiled_loss(real_result, fake_pred)
	#
	# gen_grad = gen_tape.gradient(gen_loss, gen.trainable_variables)
	# gen.optimizer.apply_gradients(zip(gen_grad, gen.trainable_variables))
	# print(disc_loss, gen_loss)
	# return disc_loss.numpy(), gen_loss.numpy()

	noise_vector = tf.random.normal(shape=(batch_size, noise_size))
	fake_numbers = gen(noise_vector)
	fake_label = tf.zeros((batch_size, 1))

	real_label = tf.ones((batch_size, 1))

	dataset = np.concatenate((fake_numbers, x_train))
	labels = np.concatenate((fake_label, real_label))

	p = np.random.permutation(len(dataset))
	x = dataset[p, ...]
	y = labels[p, ...]


	disc_loss = disc.train_on_batch(x, y)

	noise_vector = tf.random.normal(shape=(batch_size, noise_size))
	gen_loss = gan.train_on_batch(noise_vector, real_label)

	return disc_loss, gen_loss


def summarize_performance(generator, fixed_seed):
	fake_images = generator.predict(fixed_seed)
	fig = plt.figure(figsize=(12, 12))
	for i in range(25):
		plt.subplot(5, 5, i + 1)
		plt.axis('off')
		plt.imshow(fake_images[i] * 255)
	# tf.print(fake_images[i])
	plt.show()


def showplots(X, gen_lossL, disc_lossL):
	fig = plt.figure(figsize=(4, 4))
	plt.plot(X, gen_lossL, label='gen_loss')
	plt.plot(X, disc_lossL, label='disc_loss')
	plt.legend()
	plt.show()


def train(x_train, epochs, gen: tf.keras.Model, disc: tf.keras.Model, gan: tf.keras.Model, batch_size):
	x_train = np.array_split(x_train, batch_size)
	gen_lossL = []
	disc_lossL = []
	X = []

	for epoch in range(epochs):
		# progress_bar = tqdm(x_train)
		for j, image_batch in enumerate(x_train):
			disc_loss, gen_loss = train_step(image_batch, gen, disc, gan)
			gen_lossL.append(gen_loss)
			disc_lossL.append(disc_loss)
			# print(disc_loss, gen_loss)
			X.append(j + epoch * batch_size)

			showplots(X, gen_lossL, disc_lossL)
			summarize_performance(gen, tf.random.normal(shape=(25, noise_size)))

		# if epoch % 12 == 0 and epoch != 0:
		# 	disc_lr = float(tf.keras.backend.get_value(disc.optimizer.lr))
		# 	gan_lr = float(tf.keras.backend.get_value(gan.optimizer.lr))
		#
		# 	tf.keras.backend.set_value(disc.optimizer.lr, disc_lr/5)
		# 	tf.keras.backend.set_value(gan.optimizer.lr, gan_lr/5)


gen = Generator(noise_size)

disc = Discriminator((28, 28))
disc_loss = tf.keras.losses.BinaryCrossentropy()
disc_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
disc.compile(loss=disc_loss, optimizer=disc_opt)

gan = Gan(gen, disc)
gan_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
gan.compile(loss='binary_crossentropy', optimizer=gan_opt)

# x_train = x_train[:100]
train(x_train, 100, gen, disc, gan, batch_size=32)
