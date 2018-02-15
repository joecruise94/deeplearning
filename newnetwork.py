import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
# Do not import other packages

class FullyConnectedNetwork(object):
    """
    Abstraction of a Fully Connected Network.
    Stores parameters, activations, cached values.
    You can add more functions in this class, and also modify inputs and outputs of each function

    """

    def __init__(self, layer_dim, lambd=0):
        """
        layer_dim: List containing layer dimensions.

        Code:
        Initialize weight and biases for each layer
        """
        self.W = []
        self.b = []
        self.layercnt = len(layer_dim)
        for i in range(len(layer_dim) - 1):
            self.b.append(np.random.uniform(-0.01,0.01,layer_dim[i + 1]).reshape(layer_dim[i + 1],1))
            self.W.append(np.random.uniform(-0.01,0.01,layer_dim[i + 1]*layer_dim[i]).reshape(layer_dim[i + 1],layer_dim[i]))

        # for w in self.W:
        #     print (w)
        self.bestW = None
        self.bestb = None
        self.lambd = lambd

    def feedforward(self, X):
        """
        Expected Functionality:
        Returns output of the neural network for input X. Also returns cache, which contains outputs of
        intermediate layers which would be useful during backprop.

        """
        pass
        # for layers except last layer
        # affineForward
        # relu_forward
        # cache = [X]
        cache = []
        y = X
        for i in range(self.layercnt - 2):
            y, cachel = self.affineForward(y, self.W[i], self.b[i])
            #print ("The", i, "Layer output is:", y)
            #print ("y",y)
            y = self.relu_forward(y)
            #print("The", i, "Layer relu output is:", y)
            #print ("reluy",y)
            cache.append(cachel)

        # last layer
        # affineForward
        y, cachel = self.affineForward(y, self.W[-1], self.b[-1])
        #print (y)
        #print("The last Layer output is:", y)
        cache.append(cachel)
        return y, cache

    def loss_function(self, At, Y):
        """
        At is the output of the last layer, returned by feedforward.
        Y contains true labels for this batch.
        this function takes softmax the last layer's output and calculates loss.
        the gradient of loss with respect to the activations of the last layer are also returned by this function.

        """
        S = At.shape[1]
        constant_shift = np.max(At, axis=0)
        At -= constant_shift
        At = np.exp(At)
        At /= np.sum(At, axis=0)
        logy = -np.log(At)
        loss = np.sum(logy[Y[0, i], i] for i in range(S))
        At[Y, np.arange(S)] -= 1
        dL = At

        #for i in range(len(dL[0])):
        #  print (dL[:,i])
        #print ("dL", dL)
        return loss, dL
        pass
        # softmax: e^a(i) / (sum(e^a[j]) for j in all classes)
        # cross entropy loss:  -log(true class's softmax value(prediction))

        # for part2: when lambd > 0, you need to change the definition of loss accordingly

    def train(self, X, Y, max_iters=5000, batch_size=100, learning_rate=0.01, validate_every=200):
        """
        X: (3072 dimensions, 50000 examples) (Cifar train data)
        Y: (1 dimension, 50000 examples)
        lambd: the hyperparameter corresponding to L2 regularization

        Divide X, Y into train(80%) and val(20%), during training do evaluation on val set
        after every validate_every iterations and in the end use the parameters corresponding to the best
        val set to test on the Cifar test set. Print the accuracy that is calculated on the val set during
        training. Also print the final test accuracy. Ensure that these printed values can be seen in the .ipynb file you
        submit.

        Expected Functionality:
        This function will call functions feedforward, backprop and update_params.
        Also, evaluate on the validation set for tuning the hyperparameters.
        """
        # X = X.astype('float64')
        # me = np.mean(X, axis=0)
        # std = np.std(X, axis=0)
        # X -= me
        # X /= std
        trainx = X[:, 0:40000]
        valx = X[:, 40000:]
        trainy = Y[:, 0:40000]
        valy = Y[:, 40000:]
        bestacc = 0
        for i in range(max_iters):
            tx, ty = self.get_batch(trainx, trainy, batch_size)
            At, cache = self.feedforward(tx)
            loss, dAct = self.loss_function(At, ty)
            #print("The residual of last layer input is", dAct)
            d_nabla = self.backprop(0, cache, dAct)
            #print ("len of dw",len(d_nabla['w']))
            # for j in range(1,len(d_nabla['w'])+1):
            #     print ("The last", j, "layer, dw is:", d_nabla['w'][-j])
            #     print ("The last", j, "layer, db is:", d_nabla['b'][-j])
            if (i and i%4200==0):
                learning_rate = learning_rate*0.85
            self.updateParameters(d_nabla, learning_rate)
            acc = self.evaluate(valx, valy)
            if acc > bestacc:
                self.bestW = self.W
                self.bestb = self.b
            print("iteration", i, ": The accuracy is:", acc, " Loss is:", loss)
        pass

        # for some iterations
        # get current batch
        # feedforward
        # loss_function
        # backprop
        # updateParameters

    def test(self, X_test, Y_test):
        self.W = self.bestW
        self.b = self.bestb
        yout, _ = self.feedforward(X_test)
        test_res = [(np.argmax(yout, axis=0))]
        # print (Y_out.shape)
        assert test_res.shape == Y_test.shape
        tmp = test_res - Y_test
        cnt = 0
        for t in tmp[0]:
            # print (t)
            if t == 0:
                cnt += 1
        # print ("cnt",cnt)
        return cnt / len(Y_test[0])

    def affineForward(self, A, W, b):
        """
        Expected Functionality:
        Forward pass for the affine layer.
        :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
        the number of samples
        :returns: the affine product WA + b, along with the cache required for the backward pass
        """
        cachel = (A, W, b)
        return np.dot(W, A) + b, cachel
        pass

    def affineBackward(self, dA_prev, cache):
        """
        Expected Functionality:
        Backward pass for the affine layer.
        dA_prev: gradient from the next layer.
        cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """
        A, W, b = cache
        S = A.shape[1]
        dA = np.dot(W.T, dA_prev)
        dW = np.dot(dA_prev, A.T) / S
        db = (np.sum(dA_prev, axis=1) / S).reshape(dA_prev.shape[0],1)
        #print ("dwshape",dW.shape)
        return dA, dW, db
        pass

    def relu_forward(self, X):
        """
        Expected Functionality:
        Forward pass of relu activation

        returns (A, cache)
        """
        X[X <= 0] = 0
        return X
        pass

    def relu_backward(self, dx, cached_x):
        """
        Expected Functionality:
        backward pass for relu activation
        """
        dx[cached_x <= 0] = 0
        return dx
        pass

    def get_batch(self, X, Y, batch_size):
        """
        Expected Functionality:
        given the full training data (X, Y), return batches for each iteration of forward and backward prop.
        """
        n = len(Y[0])
        t = np.random.randint(n - batch_size - 1)
        tx = X[:, t:t + batch_size]
        ty = Y[:, t:t + batch_size]
        return (tx, ty)
        pass

    def backprop(self, loss, cache, dAct):
        """
        Expected Functionality:
        returns gradients for all parameters in the network.
        dAct is the gradient of loss with respect to the output of final layer of the network.
        """
        #print ("len cache",len(cache))
        #print ("cache shape")
        # for arr in cache:
        #     print (arr[0].shape)
        Nabla_W = []
        Nabla_b = []
        for i in range(self.layercnt - 1):
            tmpcache = cache[-i - 1]

            if i == 0:
                # print("i:", i)
                # print("shape of dAct", dAct.shape)
                dAct, tmpw, tmpb = self.affineBackward(dAct, tmpcache)
                #print("shape of dAct", dAct.shape)
                # print("The last", i+1, "layer, dw is:", tmpw)
                # print("The last", i + 1, "layer, dAct is:", tmpw)
                Nabla_W.append(tmpw)
                Nabla_b.append(tmpb)
                continue

            # print ("i:",i)
            # print ("shape of dAct", dAct.shape)
            # print ("shape of A", cache[-i][0].shape)
            dAct = self.relu_backward(dAct, cache[-i][0])
            dAct, tmpw, tmpb = self.affineBackward(dAct, tmpcache)
            # print("The last", i + 1, "layer, dw is:", tmpw)
            # print("The last", i + 1, "layer, dAct is:", tmpw)
            Nabla_W.append(tmpw)
            Nabla_b.append(tmpb)

        # print ("b",len(Nabla_b))
        # print ("W",len(Nabla_W))
        d_nabla = {}
        d_nabla['w'] = list(reversed(Nabla_W))
        d_nabla['b'] = list(reversed(Nabla_b))
        # for bb in d_nabla['b']:
        #     print ("shape of db", bb.shape)
        return d_nabla
        # last layer
        # affineBackward
        # set dW[last layer], db[last layer]

        # for layers except last layer
        # relu_Backward
        # affineBackward
        # set dW[layer], db[layer]

        # for part2: lambd > 0, you need to change the definition accordingly

    def updateParameters(self, gradients, learning_rate, lambd=0):
        """
        Expected Functionality:
        use gradients returned by backprop to update the parameters.
        """
        cnt = 0
        for db, dw in zip(gradients['b'], gradients['w']):
            # print ("The last",cnt+1,"layer, the shape of db", db.shape)
            # print("The last", cnt + 1, "layer, the shape of dw", dw.shape)
            assert (np.shape(self.W[cnt]) == np.shape(dw))
            self.W[cnt] -= learning_rate * dw
            self.b[cnt] -= learning_rate * db
            cnt += 1
        pass

    def evaluate(self, X_test, Y_test):
        '''
        X: X_test (3472 dimensions, 10000 examples)
        Y: Y_test (1 dimension, 10000 examples)

        Expected Functionality:
        print accuracy on test set
        '''
        yout, _ = self.feedforward(X_test)
        test_res = [(np.argmax(yout, axis=0))]
        #print (Y_out.shape)
        tmp = test_res - Y_test
        cnt = 0
        for t in tmp[0]:
            # print (t)
            if t == 0:
                cnt += 1
        # print ("cnt",cnt)
        return cnt/len(Y_test[0])
        #assert Y_out.shape == Y_test.shape
        #return sum(int(x == y) for x, y in zip(Y_out, Y_test[0])) / len(Y_test[0])
        pass



class Loader:

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_train_data(self):
        '''
        loads training data: 50,000 examples with 3072 features
        '''
        X_train = None
        Y_train = None
        for i in range(1, 6):
            pickleFile = self.unpickle('cifar-10-batches-py/data_batch_{}'.format(i))
            dataX = pickleFile[b'data']
            dataY = pickleFile[b'labels']
            if type(X_train) is np.ndarray:
                X_train = np.concatenate((X_train, dataX))
                Y_train = np.concatenate((Y_train, dataY))
            else:
                X_train = dataX
                Y_train = dataY

        Y_train = Y_train.reshape(Y_train.shape[0], 1)

        return X_train.T, Y_train.T

    def load_test_data(self):
        '''
        loads testing data: 10,000 examples with 3072 features
        '''
        X_test = None
        Y_test = None
        pickleFile = self.unpickle('cifar-10-batches-py/test_batch')
        dataX = pickleFile[b'data']
        dataY = pickleFile[b'labels']
        if type(X_test) is np.ndarray:
            X_test = np.concatenate((X_test, dataX))
            Y_test = np.concatenate((Y_test, dataY))
        else:
            X_test = np.array(dataX)
            Y_test = np.array(dataY)

        Y_test = Y_test.reshape(Y_test.shape[0], 1)

        return X_test.T, Y_test.T


X_train,Y_train = Loader().load_train_data()
X_test, Y_test = Loader().load_test_data()

print("X_Train: {} -> {} examples, {} features".format(X_train.shape, X_train.shape[1], X_train.shape[0]))
print("Y_Train: {} -> {} examples, {} features".format(Y_train.shape, Y_train.shape[1], Y_train.shape[0]))
print("X_Test: {} -> {} examples, {} features".format(X_test.shape, X_test.shape[1], X_test.shape[0]))
print("Y_Test: {} -> {} examples, {} features".format(Y_test.shape, Y_test.shape[1], Y_test.shape[0]))


layer_dimensions = [3072,200,50, 10]  # including the input and output layers
# 3072 is the input feature size, 10 is the number of outputs in the final layer
FCN = FullyConnectedNetwork(layer_dimensions, lambd=0)
FCN.train(X_train, Y_train, max_iters=100000, batch_size=200, learning_rate=0.0001,validate_every=200)
# lambd, the L2 regularization penalty hyperparamter will be 0 for this part
y_predicted = FCN.test(X_test, Y_test)  # print accuracy on test set
