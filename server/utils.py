import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
# from sklearn.metrics import plot_confusion_matrix

def separate_training_data(data, label, percent):
	length = data.shape[0]
	amount = int(length * (percent / 100))
	test_indexes = np.random.randint(length, size=amount)

	test_data = data[test_indexes]
	test_label = label[test_indexes]

	train_data = np.delete(data, test_indexes, axis=0)
	train_label = np.delete(label, test_indexes)

	return test_data, test_label, train_data, train_label


def accuracy_score(preds, y):
	return np.mean(preds == y)


def confusion_matrix(classifier, test_label, class_names, accuracy, showAnnot):
	classifier = np.round(classifier, decimals=2)
	df_cm = pd.DataFrame(classifier, columns=[np.unique(class_names)], index=[class_names[i] for i in test_label])
	plt.figure(figsize=(10, 7))
	sn.heatmap(df_cm, annot=showAnnot)
	plt.title('Accuracy score: ' + accuracy.astype(str))
	plt.xlabel('Predicted type')
	plt.ylabel('Actual type')
	plt.show()

