import idx2numpy
import numpy as np
from Files import *
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    trainImages = idx2numpy.convert_from_file('train-images-idx3-ubyte')
    trainLabels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
    testImages = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
    testLabels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')

    training_data = np.array((trainImages[:50000],trainLabels[:50000]))
    validation_data = np.array((trainImages[50000:],trainLabels[50000:]))
    test_data = np.array((testImages,testLabels))
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    mndata = MNIST("")
    mndata.load_training()
    mndata.load_testing()
    mndata.train_images = mndata.train_images
    mndata.test_images = np.array(mndata.test_images)
    training_inputs = [np.reshape(x, (784)).astype(np.float32)/256 for x in mndata.train_images[:50000]]
    training_results = mndata.train_labels[:50000]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)).astype(np.float32)/256 for x in mndata.train_images[50000:]]
    validation_data = zip(validation_inputs, mndata.train_labels[50000:])
    test_inputs = [np.reshape(x, (784, 1)).astype(np.float32)/256 for x in mndata.test_images]
    test_data = zip(test_inputs, mndata.test_labels)
    return (training_inputs, training_results)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e