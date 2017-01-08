from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.relu):
        # ReLU = max(0,z)
        # for the activation of the hidden layer, I put Rectified Unit as the standard, as it is
        # easy to optimize due to their similarity to the linear unit.
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        #if would like to have a linear function then just put identity as activation
        self.output = activation(lin_output)

        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

        A multilayer perceptron is a feedforward artificial neural network model
        that has one layer or more of hidden units and nonlinear activations.
        Intermediate layers usually have as activation function tanh or the
        sigmoid function (defined here by a ``HiddenLayer`` class)  while the
        top layer is a softmax layer (defined here by a ``LogisticRegression``
        class).
        """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: array
        :param n_hidden: number of hidden units and their sizes

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayers = []
        k = 0
        for i in range(len(n_hidden)):
            self.hiddenLayers.append(HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden[k],
                activation=T.cos
            ))
            input = self.hiddenLayers[-1].output
            n_in = n_hidden[k]
            k += 1

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=n_hidden[-1],
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        hLW = 0
        hLW2 = 0
        hL_Params = self.logRegressionLayer.params
        for i in self.hiddenLayers:
            hLW += abs(i.W.sum())
            hLW2 += (i.W**2).sum()
            hL_Params += i.params

        self.L1 = (
            hLW+abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            hLW2 +
            (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = hL_Params
        # end-snippet-3

        # keep track of model input
        self.input = input
class mlp_model(object):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    def __init__ (self,datasets,learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001,n_in=28*28, n_hidden = [100, 20, 20, 20, 20],n_out=10,batch_size=20):

        self.batch_size = batch_size
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        self.rng = numpy.random.RandomState(1234)

        # construct the MLP class
        self.classifier = MLP(
            rng=self.rng,
            input=self.x,
            n_in=n_in,
            n_hidden=n_hidden,
            n_out=n_out
        )

        # start-snippet-4
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        self.cost = (
            self.classifier.negative_log_likelihood(self.y)
            + L1_reg * self.classifier.L1
            + L2_reg * self.classifier.L2_sqr
        )
        # end-snippet-4

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        self.test_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: self.test_set_x[self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            }
        )
        self.validate_model = theano.function(
            inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: self.valid_set_x[self.index * self.batch_size:(self.index + 1) * self.batch_size],
                self.y: self.valid_set_y[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            }
        )

        # start-snippet-5
        # compute the gradient of cost with respect to theta (sorted in params)
        # the resulting gradients will be stored in a list gparams
        self.gparams = [T.grad(self.cost, param) for param in self.classifier.params]
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs

        self.updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.classifier.params, self.gparams)
            ]
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        self.train_model = theano.function(
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: self.train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
    def train_mlp(self, n_epochs=1000):
        ###############
        # TRAIN MODEL #
        ###############
        print('... training')


        # compute number of minibatches for training, validation and testing
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] // self.batch_size
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] // self.batch_size
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] // self.batch_size

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = self.train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if (
                                    this_validation_loss < best_validation_loss *
                                    improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [self.test_model(i) for i
                                       in range(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=[100,20,20,20,20]):
    """
     Demonstrate stochastic gradient descent optimization for a multilayer
     perceptron

     This is demonstrated on MNIST.

     :type learning_rate: float
     :param learning_rate: learning rate used (factor for the stochastic
     gradient

     :type L1_reg: float
     :param L1_reg: L1-norm's weight when added to the cost (see
     regularization)

     :type L2_reg: float
     :param L2_reg: L2-norm's weight when added to the cost (see
     regularization)

     :type n_epochs: int
     :param n_epochs: maximal number of epochs to run the optimizer

     :type dataset: string
     :param dataset: the path of the MNIST dataset file from
                  http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


    """
    datasets = load_data(dataset)
    model = mlp_model(datasets=datasets)
    model.train_mlp()

if __name__ == '__main__':
    test_mlp()
    #dataset = 'mnist.pkl.gz'
    #datasets = load_data(dataset)
    #mlp = mlp_model()
    #mlp.train_mlp(datasets=datasets, n_epochs=1000, batch_size=20)


