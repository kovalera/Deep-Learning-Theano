from theano import function, shared
import theano.tensor as T
import numpy as np
import theano
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.on_unused_input='warn'
class Linear_Regression():
    def __init__(self,input, features_num):

        self.w = shared(value = np.zeros((features_num,),dtype=theano.config.floatX),
                        name = 'w',borrow=True)
        # First feature of the array is always one, to substitute for the use of b
        self.output = T.dot(input,self.w)
        self.input = input


    def MSE(self,y):
        return T.mean(T.power(self.output-y,2))

    def predict(self,x):

        v = T.dmatrix()
        p = T.dot(v,self.w)
        f = function([v],p)
        #printing the function in an image file
        #theano.printing.pydotprint(f, 'gemv_case2', format='png')
        return f(x)


def Train_LR(x_train, y_train,epsilon,n_epochs):
    #Creation of the model
    # DON'T DO VECTOR if you are doing BATCH, do matrix!!
    x = T.matrix('x')
    y = T.vector('y')

    LR = Linear_Regression(x,len(x_train[0]))

    cost = LR.MSE(y)

    #Calculation of the Gradient of the cost
    dw = T.grad(cost=cost,wrt=LR.w)

    updates = [(LR.w,LR.w-epsilon*dw)]#,(LR.b,LR.b-epsilon*db)]
    index = T.lscalar()
    train_model = function(inputs=[],outputs = cost,updates=updates,givens={x: x_train[:],y:y_train[:]})

    #Running of the model
    for i in range(n_epochs):
        #for x_t,y_t in zip(x_train,y_train):
        #    trained_cost = train_model(x_t,y_t)
        trained_cost = train_model()
        print trained_cost
    return LR


if __name__=="__main__":
    x = np.array([[2,1], [3,2], [4,3], [5,4], [6,5]], dtype=theano.config.floatX)
    #x_train = np.ones((len(x),len(x[0])+1))
    #x_train[:,1:] = x
    # same as above but prettier
    x_train = np.hstack((np.ones((x.shape[0], 1), dtype=x.dtype),x))
    y_train = np.array([2.3, 4.1, 5.8, 8.1, 10], dtype=theano.config.floatX)
    LR = Train_LR(x_train,y_train,0.01,200)
    x = np.array([[2,1], [3,2], [4,3], [5,4], [6,5], [7,6], [8,7], [9,8], [10,9], [11,10]])
    x_test = np.hstack((np.ones((x.shape[0], 1), dtype=x.dtype), x))
    print LR.predict(x_test)

