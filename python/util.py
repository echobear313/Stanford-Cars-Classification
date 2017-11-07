import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np


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


if __name__ == "__main__":
	data = loadmat("/Users/HZzone/Desktop/Stanford-Cars-Classification/data/cars_train_annos.mat", "annotations")
	print(data)

	data = loadmat("/Users/HZzone/Desktop/Stanford-Cars-Classification/data/cars_test_annos_withlabels.mat", "annotations")
	print(data)
