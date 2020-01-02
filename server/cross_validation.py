import numpy as np


def create_buckets(data, label_list, n):
	data, label_list = shuffle(data, label_list)

	b_labels = np.array(np.array_split(label_list, n))
	b_data = np.array(np.array_split(data, n))

	return b_data, b_labels


def shuffle(data, label_list):
	indexes = np.arange(label_list.shape[0]) # creates an array from 0 to arr-length
	np.random.shuffle(indexes)
	return data[indexes], label_list[indexes]

