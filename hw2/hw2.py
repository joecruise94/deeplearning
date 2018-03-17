import numpy as np
import pickle
from deeplearning.hw2.layer import *
from deeplearning.hw2 import optimizer
from deeplearning.hw2.fast_layer import conv_forward_fast, conv_backward_fast
import matplotlib.pyplot as plt
from time import time

class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.verbose = kwargs.pop('verbose', True)
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)

        self.print_every = kwargs.pop('print_every', 10)
        if not hasattr(optimizer, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in kwargs.keys())
            raise ValueError('Unrecognized arguments %s' % extra)
        self.update_rule = getattr(optimizer, self.update_rule)
        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)  ###################
        self.loss_history.append(loss)

        # Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def check_accuracy(self, X, y, num_samples=None, batch_size=200):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        #print ("check accuracy")
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        #print ("N", N, " barch_size", batch_size, " num_batches", num_batches)
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])  #########################
            #print ("scores shape", scores.shape)
            #print (np.argmax(scores, axis=1).shape)
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(int(num_train / self.batch_size), 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)
        #print (num_train)
        #print (iterations_per_epoch)

        for t in range(num_iterations):
            self._step()
            if self.verbose and t % self.print_every == 0:
                print ('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))
            #print ("success 1")
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                                num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params


    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
          'model': self.model,
          'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
          'optim_config': self.optim_config,
          'batch_size': self.batch_size,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'train_acc_history': self.train_acc_history,
          'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)


class BasicConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, reg=0.0, dropout=0, use_batchnormal=False):
        """
        Initialize a network
        :param input_dim:
        :param num_filters:
        :param filter_size:
        :param hidden_dim:
        :param num_classes:
        :param weight_scale:
        :param reg:
        """
        self.params = {}
        self.reg = reg
        self.dropout = dropout
        self.use_dropout = False
        self.use_batchnormal=use_batchnormal
        C, H, W = input_dim
        wscale = 2e-2

        self.params['W1'] = wscale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = wscale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
        self.params['b2'] = np.zeros(num_filters)
        self.params['W3'] = wscale * np.random.normal(0, np.sqrt(2.0 / (int((H / 2) * (W / 2)) * num_filters)), size=(int((H / 2) * (W / 2)) * num_filters, hidden_dim))
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = wscale * np.random.normal(0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, num_classes))
        self.params['b4'] = np.zeros(num_classes)

        if dropout > 0:
            self.use_dropout = True
            self.dropout_param = {'mode': 'train', 'p': dropout}



    def loss(self, X, y=None):
        """
        Get the loss
        :param X:
        :param y:
        :return:
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        filter_size1 = W1.shape[2]
        filter_size2 = W2.shape[2]
        conv_param1 = {'stride': 1, 'pad': int((filter_size1 - 1) / 2)}
        conv_param2 = {'stride': 1, 'pad': int((filter_size2 - 1) / 2)}
        pool1_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        a, conv1_cache = conv_forward_fast(X, W1, b1, conv_param1)
        s, relu1_cache = relu_forward(a)
        out1, pool1_cache = max_pool_forward(s, pool1_param)
        a2, conv2_cache = conv_forward_fast(out1, W2, b2, conv_param2)
        out2, relu2_cache = relu_forward(a2)
        af_out3, af3_cache = affineForward(out2, W3, b3)
        relu_out3, relu3_cache = relu_forward(af_out3)
        scores, af4_cache =affineForward(relu_out3, W4, b4)

        if y is None:
            return scores

        loss, dout = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2) + np.sum(self.params['W3'] ** 2))+np.sum(self.params['W4'] ** 2)
        grads = {}
        dX4, grads['W4'], grads['b4'] = affineBackward(dout, af4_cache)
        dX3 = relu_backward(dX4, relu3_cache)
        dX3, grads['W3'], grads['b3'] = affineBackward(dX3, af3_cache)
        dX2 = relu_backward(dX3, relu2_cache)
        dX2, grads['W2'], grads['b2'] = conv_backward_fast(dX2, conv2_cache)
        dX2_pool = max_pool_backward(dX2, pool1_cache)
        dX1 = relu_backward(dX2_pool, relu1_cache)
        dX1, grads['W1'], grads['b1'] = conv_backward_fast(dX1, conv1_cache)

        grads['W3'] = grads['W3'] + self.reg * self.params['W3']
        grads['W2'] = grads['W2'] + self.reg * self.params['W2']
        grads['W1'] = grads['W1'] + self.reg * self.params['W1']
        grads['W4'] = grads['W4'] + self.reg * self.params['W4']
        return loss, grads






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
            pickleFile = self.unpickle('../../cifar-10-batches-py/data_batch_{}'.format(i))
            dataX = pickleFile[b'data']
            dataY = pickleFile[b'labels']
            if type(X_train) is np.ndarray:
                X_train = np.concatenate((X_train, dataX))
                Y_train = np.concatenate((Y_train, dataY))
            else:
                X_train = dataX
                Y_train = dataY

        X_train = X_train.reshape((X_train.shape[0],3,32,32)).astype("float")
        Y_train = Y_train.astype("float")


        return X_train, Y_train

    def load_test_data(self):
        '''
        loads testing data: 10,000 examples with 3072 features
        '''
        X_test = None
        Y_test = None
        pickleFile = self.unpickle('../../cifar-10-batches-py/test_batch')
        dataX = pickleFile[b'data']
        dataY = pickleFile[b'labels']
        if type(X_test) is np.ndarray:
            X_test = np.concatenate((X_test, dataX))
            Y_test = np.concatenate((Y_test, dataY))
        else:
            X_test = np.array(dataX)
            Y_test = np.array(dataY)

        X_test = X_test.reshape((X_test.shape[0],3,32,32)).astype("float")
        Y_test = Y_test.astype("float")

        return X_test, Y_test

X_train,Y_train = Loader().load_train_data()
X_test, Y_test = Loader().load_test_data()
X_val = X_train[40000:]
Y_val = Y_train[40000:]
X_train = X_train[:40000]
Y_train = Y_train[:40000]

###Normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
#print (X_train.shape)
data = {
      'X_train': X_train, 'y_train': Y_train,
      'X_val': X_val, 'y_val': Y_val,
      'X_test': X_test, 'y_test': Y_test,
}

for k, v in data.items():
  print ('%s: ' % k, v.shape)

"""
For training
"""



model = BasicConvNet( hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=100,
                update_rule='sgd',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

