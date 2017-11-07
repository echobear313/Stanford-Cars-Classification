import cv2
import util
import numpy as np
import os
import logging
import lmdb
import sys
sys.path.insert(0, "/home/hzzone/caffe/python")
import caffe


train_folder = "../data/cars_train"
test_folder = "../data/cars_test"
IMAGE_SIZE = 227
PIXEL_MEAN = 0.40651412216

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S')

def generate(source, mat_source, data_save_path, label_save_path):
	data = util.loadmat(mat_source, "annotations")
	labels = data[:, -2] - 1
	counts = data.shape[0]
	all_data = np.zeros((counts, 3, IMAGE_SIZE, IMAGE_SIZE))
	for index in range(counts):
		file_name = data[index][-1]
		path = os.path.join(source, file_name)
		im = cv2.imread(path)
		x1 = data[index][0]
		y1 = data[index][1]
		x2 = data[index][2]
		y2 = data[index][3]
		im = im[y1:y2, x1:x2]
		im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
		im = np.transpose(im, (2, 0, 1))
		# all_data.astype(np.float)
		all_data[index][...] = im*0.00390625 - PIXEL_MEAN
		logging.debug("%s %s" % (index, path))
	# data_label = np.array((all_data, labels))
	np.save(data_save_path, all_data)
	np.save(label_save_path, labels)

def generate_lmdb(save_path, data_source, label_source, mat_source):
	env = lmdb.Environment(save_path, map_size=int(1e12))
	data = np.load(data_source)
	labels = np.load(label_source)
	print(labels.dtype)
	print(data.dtype)
	with env.begin(write=True) as txn:
		for index in range(data.shape[0]):
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = 3
			datum.height = IMAGE_SIZE
			datum.width = IMAGE_SIZE
			datum.data = data[index].tobytes()
			datum.label = labels[index]
			str_id = "%8d" % index
			txn.put(str_id, datum.SerializeToString())
			logging.debug("%s %s" % (str_id, labels[index]))
	logging.debug("%s complete" % save_path)

if __name__ == "__main__":
	# generate(train_folder, "../data/cars_train_annos.mat", "../data/train_data.npy", "../data/train_label.npy")
	# generate(test_folder, "../data/cars_test_annos_withlabels.mat", "../data/test_data.npy", "../data/test_label.npy")
	generate_lmdb("/home/hzzone/1tb/cars-data/train_lmdb", "../data/train_data.npy", "../data/train_label.npy", "../data/cars_train_annos.mat")
	generate_lmdb("/home/hzzone/1tb/cars-data/test_lmdb", "../data/test_data.npy", "../data/test_label.npy", "../data/cars_test_annos_withlabels.mat")
