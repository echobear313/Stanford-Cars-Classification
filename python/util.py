import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import preprocess
from functools import reduce


'''
cars_meta.mat: class_names
cars_test_annos.mat, cars_train_annos.mat, cars_test_annos_withlabels.mat:
				annotations
'''
def loadmat(source, coloum):
	data = sio.loadmat(source)
	data = data[coloum]
	data = np.array(data[0].tolist())
	np.reshape(data, (-1, 6))
	data = np.transpose(data, (2, 0, 1))
	data = data[0]
	data[:, 0:5].astype(int)
	return data

def show(im):
	plt.imshow(im)
	plt.show()

def calculate_mean():
	train_data = preprocess.generate("../data/cars_train", "../data/cars_train_annos.mat")
	train_shape = train_data.shape
	train_sum = np.sum(train_data)
	del train_data
	test_data = preprocess.generate("../data/cars_test", "../data/cars_test_annos_withlabels.mat")
	test_shape = test_data.shape
	test_sum = np.sum(test_data)
	del test_data
	# data = np.vstack((train_data, test_data))
	pixel_sum = train_sum + test_sum
	pixel_counts = reduce(lambda x, y: x * y, test_shape) + reduce(lambda x, y: x * y, train_shape)
	return pixel_sum/pixel_counts

if __name__ == "__main__":
	# data = loadmat("/Users/HZzone/Desktop/Stanford-Cars-Classification/data/cars_train_annos.mat", "annotations")
	# print(data)
	#
	# data = loadmat("/Users/HZzone/Desktop/Stanford-Cars-Classification/data/cars_test_annos_withlabels.mat", "annotations")
	# print(data)
	print(calculate_mean())
