import numpy as np

def get_data(source):

	# Create numpy array data from blogdata.txt
	title = np.genfromtxt(source, delimiter=',', dtype=str)[0] # Titles of the data columns
	data = np.genfromtxt(source, delimiter=',', skip_header=1)[:, 0:-1] # The integer-data of the source file
	data_type = np.genfromtxt(source, delimiter=',', dtype=str, usecols=4)[1:] # The name of the current row
	return title, data, data_type


def get_label(original):
	new = np.zeros(shape=original.shape)
	types = np.unique(original)
	for i, unique_type in enumerate(types):
		new[np.where(original == unique_type)[0]] = i

	return new.astype(int), types



