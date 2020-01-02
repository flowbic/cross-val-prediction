import numpy as np
from algorithm import *


def crossval_predict(data, label_list, n):
	data_buckets, label_buckets = create_buckets(data, label_list, n)
	# print(label_buckets)
	tot_score = 0
	for i, bucket in enumerate(data_buckets):
		test_data = bucket
		test_labels = label_buckets[i]
		train_labels = np.concatenate(np.delete(label_buckets, i, axis=0))
		train_data = np.concatenate(np.delete(data_buckets, i, axis=0))

		fit(train_data, train_labels)
		norm_array = predict(test_data)
		result = np.argmax(norm_array, axis=1)
		tot_score += accuracy_score(result, test_labels)

	return tot_score / n


def create_buckets(data, label_list, n):
	data, label_list = shuffle(data, label_list)

	b_labels = np.array(np.array_split(label_list, n))
	b_data = np.array(np.array_split(data, n))

	return b_data, b_labels


def shuffle(data, label_list):
	indexes = np.arange(label_list.shape[0]) # creates an array from 0 to arr-length
	np.random.shuffle(indexes)
	return data[indexes], label_list[indexes]

