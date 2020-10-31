import pickle
import numpy as np

"""This script implements the functions for reading data.
"""


def loadData(data_dir):
    """Load the CIFAR-10 dataset.

	Args:
		data_dir: A string. The directory where data batches
			are stored.

	Returns:
		x_train: An numpy array of shape [50000, 3072].
			(dtype=np.float32)
		y_train: An numpy array of shape [50000,].
			(dtype=np.int32)
		x_test: An numpy array of shape [10000, 3072].
			(dtype=np.float32)
		y_test: An numpy array of shape [10000,].
			(dtype=np.int32)
	"""
    x_train = None
    y_train = None
    for i in range(1, 6):
        with open("%s/data_batch_%d" % (data_dir, i), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            if x_train is None:
                x_train = dict[b'data']
                y_train = dict[b'labels']
            else:
                x_train = np.concatenate((x_train, dict[b'data']))
                y_train = np.concatenate((y_train, dict[b'labels']))
    with open("%s/test_batch" % data_dir, 'rb') as fo:
        x_test = dict[b'data']
        y_test = dict[b'labels']

    x_train = x_train.reshape((50000, 3, 32, 32))
    x_test = x_test.reshape((10000, 3, 32, 32))

    return x_train, y_train, x_test, y_test


def trainValidSplit(x_train, y_train, split_index=45000):
    """Split the original training data into a new training dataset
	and a validation dataset.

	Args:
		x_train: An array of shape [50000, 3072].
		y_train: An array of shape [50000,].
		split_index: An integer.

	Returns:
		x_train_new: An array of shape [split_index, 3072].
		y_train_new: An array of shape [split_index,].
		x_valid: An array of shape [50000-split_index, 3072].
		y_valid: An array of shape [50000-split_index,].
	"""
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid

def localNorm(data):
    data = [(d-np.mean(d)) / np.std(d) for d in data ]
    return np.asarray(data)

def parse_record(record, training=True):
	"""Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	# image = preprocess_image(image, training)

	return image