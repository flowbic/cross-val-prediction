from manage_files import *
from utils import *
from cross_validation import *
from algorithm import *
import numpy as np


def run(path, n):
	title, data, types = get_data(path)
	label_list, unique_types = get_label(types)
	data_buckets, label_buckets = create_buckets(data, label_list, n)
	# print(label_buckets)

	score = crossval_predict(data, label_list, n)
	print(n + '-fold Cross validation, average: ' + score)
	print(score)

	fit(data, label_list)
	norm_array = predict(data)
	result = np.argmax(norm_array, axis=1)
	print(norm_array)
	print(accuracy_score(result, label_list))

	confusion_matrix(norm_array, label_list, unique_types, accuracy_score(result, label_list), True)


run('Iris/iris.csv', 4)
run('banknote_authentication.csv', 4)


def iris():
	title, data, types = get_data('Iris/iris.csv')
	label_list, unique_types = get_label(types)

	# Separates data in two categories, test and train data. Test data will be as large as the input percentage of data
	test_data, test_labels, train_data, train_labels = separate_training_data(data, label_list, 10)

	fit(train_data, train_labels)
	norm_array = predict(test_data)
	result = np.argmax(norm_array, axis=1)
	print(norm_array)
	print(accuracy_score(result, test_labels))

	confusion_matrix(norm_array, test_labels, unique_types, accuracy_score(result, test_labels), True)


# iris()


def banknote():
	title, data, image_class = get_data('banknote_authentication.csv')
	label_list, unique_types = get_label(image_class)

	# Separates data in two categories, test and train data. Test data will be as large as the input percentage of data
	test_data, test_labels, train_data, train_labels = separate_training_data(data, label_list, 10)

	fit(train_data, train_labels)
	norm_array = predict(test_data)
	result = np.argmax(norm_array, axis=1)
	print(norm_array)
	print(accuracy_score(result, test_labels))

	confusion_matrix(norm_array, test_labels, unique_types, accuracy_score(result, test_labels), False)


# banknote()
