from manage_files import *
from utils import *
import numpy as np

model = []
model_labels = []


def fit(x, y):
	model.clear()
	labels = np.unique(y)
	for label in labels:
		current_x = x[np.where(y == label)[0]]
		mean = np.mean(current_x, axis=0)
		std = np.std(current_x, axis=0)
		# model.append({'mean': mean, 'std': std, 'label': label})
		model.append([mean, std])
		model_labels.append(label)


def predict(x):
	x = x
	m = np.array(model)
	mean = m[::, 0]
	std = m[::, 1]

	results = []
	for xi in x:
		pdf = (1 / np.sqrt(2 * np.pi) * std) * np.exp(-((xi - mean) ** 2 / (2 * std ** 2)))
		p = np.exp(np.sum(np.log(pdf), axis=1))  # == np.prod(pdf, axis=1)
		p_norm = p / np.sum(p)

		results.append(p_norm)

	return np.array(results)

