import tensorflow as tf
import numpy as np
import random
import ast


def generate_traindata():
	names = ["Jane", "Doug", "Spot", "Seba", "Jul", "Gab"]
	verbs = ["saw", "killed", "ate"]
	separators = ["and", ",", "."]

	data = []
	for i in range(10000):
		n_sep = random.randint(1, len(names) - 1)
		available_names = names.copy()
		available_targetname = names.copy()
		mysentence = []
		for x in range(n_sep):
			name = random.choice(available_names)
			try:
				available_names.remove(name)
			except:
				pass
			verb = random.choice(verbs)

			tmp = available_targetname.copy()
			try:
				tmp.remove(name)
			except:
				pass
			target_name = random.choice(tmp)

			mysentence.append(name)
			mysentence.append(verb)
			mysentence.append(target_name)
			if x < n_sep - 2:
				mysentence.append(",")
			elif x == n_sep - 2:
				mysentence.append("and")
			else:
				mysentence.append(".")
		data.append(mysentence)
	return data

# data = generate_traindata()
# myfile = open("trainingdata.txt", "w")
# myfile.write(str(data))
# myfile.close()
myfile = open("trainingdata.txt", "r")
data = ast.literal_eval(myfile.read())


def neural_network():
	model = tf.keras.models.Sequential()

	model.add(tf.keras.layers.Dense(64, activation='tanh'))
	model.add(tf.keras.layers.Dense(64, activation='somethinghere'))

	model.add()

class RecurrentShit:
	pass