import numpy as np


def conv_forward(x, w, b, conv_param):
    """
    Convolution forward pass
    :param x: N, C, H,W
    :param w: F, C, HH, WW
    :param b: F,
    :param stride: step
    :param pad: padding
    :return:
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    """
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out)).astype("float")

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):
                out[:, k, i, j] = np.sum(x_pad_masked * w[k, :, :, :], axis=(1, 2, 3))

    out = out + (b)[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache








def conv_backward(dout, cache):
    """
    convolution backpropagation
    :param dout: derivatives
    :param cache: (x, w, b, conv_param)
    :return: dx, dw, db
    """
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):  # compute dw
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):  # compute dx_pad
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] *
                                                                                                (dout[n, :, i, j])[:,
                                                                                                None, None, None]),
                                                                                               axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]

    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    max pooling forward
    :param x: (N, C, H, W)
    :param pool_param:
    Height,
    Width
    stride
    :return:
    """
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    out = np.zeros((N, C, H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward(dout, cache):
    """
    backward pass for max pooling
    :param dout: derivatives
    :param cache:
    :return: dx
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    HH, WW, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    H_out = int((H - HH) / stride + 1)
    W_out = int((W - WW) / stride + 1)
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_masked = x[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW]
            max_x_masked = np.max(x_masked, axis=(2, 3))
            temp_binary_mask = (x_masked == (max_x_masked)[:, :, None, None])
            dx[:, :, i * stride: i * stride + HH, j * stride: j * stride + WW] += temp_binary_mask * (dout[:, :, i, j])[
                                                                                                     :, :, None, None]
    return dx

def softmax_loss(x, y):
    """
    compute the loss and gradient
    :param x: (N, C) where x[i, j] is the score for the jth class
    for the ith input.
    :param y:shape (N,) where y[i] is the label for x[i]
    :return:
    """
    #print ("softmax_loss")
    #print (x.shape, "yshape", y.shape)
    #print (np.max(x, axis=1, keepdims=True))
    #probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs = np.exp(x)
    probs /= np.sum(probs, axis=1, keepdims=True)
    probs = probs.clip(min=10e-17)
    #print ("probshape", probs.shape)
    N = x.shape[0]
    #print (y.shape)
    if y.all() != None:
        y = y.astype("int")
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def relu_forward(X):
    """
    Expected Functionality:
    Forward pass of relu activation

    returns (A, cache)
    """
    cache = X
    X[X <= 0] = 0
    return X, cache
    pass


def relu_backward(dx, cached_x):
    """
    Expected Functionality:
    backward pass for relu activation
    """
    dx[cached_x <= 0] = 0
    return dx
    pass



def affineBackward(dout, cache):
    """
    Expected Functionality:
    Backward pass for the affine layer.
    dA_prev: gradient from the next layer.
    cache: cache returned in affineForward
    :returns dA: gradient on the input to this layer
             dW: gradient on the weights
             db: gradient on the bias
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    dx = dout.dot(w.T)
    dx = dx.reshape(*x.shape)
    dw = x_rsp.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def affineForward(x, w, b):
    """
    Expected Functionality:
    Forward pass for the affine layer.
    :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
    the number of samples
    :returns: the affine product WA + b, along with the cache required for the backward pass
    """
    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    out = x_rsp.dot(w) + b
    cache = (x, w, b)
    return out, cache





def rel_error(x, y):
    """ returns relative error """
    return np.mean(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print (ix, grad[ix])
    it.iternext() # step to next dimension

  return grad