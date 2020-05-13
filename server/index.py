from cross_validation import *
from algorithm import *
import numpy as np


def run(path, n):
	title, data, types = get_data(path)
	label_list, unique_types = get_label(types)

	score = crossval_predict(data, label_list, n)
	print(n, '-fold Cross validation, average: ', score)

	fit(data, label_list)
	norm_array = predict(data)
	result = np.argmax(norm_array, axis=1)
	print(accuracy_score(result, label_list))

	confusion_matrix(norm_array, label_list, unique_types, accuracy_score(result, label_list), False)
	print(result.shape)


run('Iris/iris.csv', 5)

run('banknote_authentication.csv', 5)
