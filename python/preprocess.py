import cv2
import util
import numpy as np
import os

train_folder = ""
test_folder = ""
IMAGE_SIZE = 227

def generate(source, mat_source):
	data = util.loadmat(mat_source)
	counts = data.shape[0]
	all_data = np.zeros((counts, IMAGE_SIZE, IMAGE_SIZE))
	for index in range(counts):
		file_name = data[index][-1]
		path = os.path.join(source, file_name)
		im = cv2.imread(path)
		x1 = data[index][1]
		y1 = data[index][2]
		x2 = data[index][3]
		y2 = data[index][4]
		im = im[y1:y2, x1:x2]
		util.show(im)
		exit()


if __name__ == "__main__":
	generate()
