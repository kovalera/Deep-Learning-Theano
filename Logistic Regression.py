import theano.tensor as T
from theano import function, shared
import theano
from mnist_loader2 import load_data_wrapper
import numpy as np
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
class Logistic_Regression():
    def __init__(self,input, features_num, outputs_num):

        self.w = shared(value = np.zeros((features_num,outputs_num),dtype=theano.config.floatX),
                        name = 'w',borrow=True)
        # First feature of the array is always one, to substitute for the use of b
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.w))

        self.predict_y = T.argmax(self.p_y_given_x)
        self.input = input


    def MSE(self,y):
        return T.mean(T.power(self.predict_y-y,2))
    def NLL(self,y):
        """
        Cost function we would like to minimize, Negative Log Likelihood
        Minimization of the negative log likelihood is taking care of the
        saturation problem that exists in the usage of sigmoids and softmax
        activation functions, since the log is cancelling the exp.

            T.arange(5) = [0,1,2,3,4]
            a[[x's,y's]] = [a[x1,y1],a[x2,y2]...]
            checks the p of the y at the place where y = 1,
            we want to maximize that
            -1/m * sum(log(p(y|x)))  ** Where y = 1
            e.g. 10th example, where is the y =1 what is p_y_given_x
            """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self,y):
        a = self.predict_y
        f = function([],a)
        print f()
        return T.mean(T.neq(self.predict_y, y))

def Train_LogR(or_train,x_train,y_train,epsilon,epochs,batch_size=1000):
    train_batches = len(x_train)/batch_size
    x = T.matrix()
    y = T.ivector()

    shared_y = theano.shared(y_train)
    shared_x = theano.shared(x_train)

    shared_y = shared_y.flatten()
    shared_y = T.cast(shared_y, 'int32')

    shared_or = theano.shared(or_train)
    shared_or = shared_or.flatten()
    shared_or = T.cast(shared_or, 'int32')

    cats = len(y_train[0])
    feats = len(x_train[0])

    model = Logistic_Regression(x,feats,cats)

    cost = model.NLL(y)
    dw = T.grad(cost=cost,wrt=model.w)
    updates = [(model.w , model.w - epsilon * dw)]

    #Gradient descent Function

    sh_y = T.iscalar()
    """
    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(sh_y),on_unused_input='warn',
        givens={
            x: shared_x[index * batch_size:(index + 1) * batch_size],
            sh_y: shared_or[index * batch_size:(index + 1) * batch_size]
        }
    )
    """
    index = T.lscalar()
    GD = function(inputs=[index],outputs=cost,updates=updates,allow_input_downcast=True,on_unused_input='warn',
                  givens={x: shared_x[index * batch_size: (index + 1) * batch_size],
                          y: shared_y[index * batch_size: (index + 1) * batch_size]})
    test_model = theano.function(
        inputs=[index],
        outputs=model.errors(y),
        givens={
            x: shared_x[index * batch_size: (index + 1) * batch_size],
            y: shared_or[index * batch_size: (index + 1) * batch_size]
        }
    )

    for i in range(epochs):
        print i
        t_cost = 0
        for ind in range(train_batches):
            e_cost = GD(ind)
        t_cost = test_model(1)
        print t_cost
def toArr(arr,leng):
    out = np.zeros((len(arr),leng))
    for i in range(len(arr)):
        out[i][arr[i]] = 1
    return out


if __name__ =="__main__":

    x_train, y_train = load_data_wrapper()
    # adding 1's to substitute for the bias variable
    x_train = np.hstack((np.ones((len(x_train), 1)), x_train))
    or_train = np.array(y_train)
    y_train = np.array(toArr(y_train,10))
    Train_LogR(or_train,x_train, y_train, 0.01, 100, 1000)

