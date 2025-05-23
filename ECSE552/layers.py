import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    #############################################################################
    nelements = w.shape[1]
    out = np.dot(x.reshape(x.shape[0], -1), w) + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    dw = np.dot(dout.T, x.reshape(x.shape[0], -1)).T
    db = np.sum(dout, axis=0)
    dx = np.dot(dout, w.T).reshape(x.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    # Hint: You can also use im2col or im2col_indices see the file im2col.py    #
    #       for further information                                             #
    #############################################################################
    # extract shapes
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # extract stride and pad
    stride, pad = conv_param["stride"], conv_param["pad"]

    # pad the input with zeros
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

    # compute shape of output data using floor division
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    # initialize the output
    out = np.zeros((N, F, H_prime, W_prime))

    # perform convolution
    for n in range(N): # iterate over the samples
        for f in range(F): # iterate over the filters
            for i in range(H_prime): # iterate over the output height
                for j in range(W_prime): # iterate over the output width
                  # defines the receptive field
                  h_start = i * stride
                  h_end = h_start + HH
                  
                  w_start = j * stride
                  w_end = w_start + WW

                  # convolve
                  out[n, f, i, j] = np.sum(x_padded[n, :, h_start:h_end, w_start:w_end] * w[f, :, :, :]) + b[f]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    # extract parameters
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # extract shapes
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_prime, W_prime = dout.shape

    # initialize gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # pad input like in forward pass and initialize gradient
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)

    # compute db according to equation 1 (see notes)
    db = np.sum(dout, axis = (0, 2, 3))

    # calculate gradients
    for n in range(N): # iterate over the samples
        for f in range(F): # iterate over the filters
            for i in range(H_prime): # iterate over the output height
                for j in range(W_prime): # iterate over the output width
                  # defines the receptive field
                  h_start = i * stride
                  h_end = h_start + HH
                  
                  w_start = j * stride
                  w_end = w_start + WW

                  # accumulate gradients
                  # equation 2 (see notes)
                  dw[f] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                  # equation 3 (see notes)
                  dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, i, j]
    
    # remove padding
    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    # extract shapes
    N, C, H, W = x.shape

    # extract parameters
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # calculate the dimensions of the output
    H_prime = (H - pool_height) // stride + 1
    W_prime = (W - pool_width) // stride + 1

    # initialize the output array
    out = np.zeros((N, C, H_prime, W_prime))

    # iterate over the samples
    for n in range(N):
      # iterate over the channels
        for c in range(C):
          # iterate over the output height
          for h in range(H_prime):
            # iterate over the output width
            for w in range(W_prime):
              # defines the receptive field
              h_start = h * stride
              h_end = h_start + pool_height
              
              w_start = w * stride
              w_end = w_start + pool_width

              # extract the receptive field and apply max pooling
              out[n, c, h, w] = np.max(x[n, c, h_start:h_end, w_start:w_end])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    # retrieve the input and the pool parameters
    x, pool_param = cache
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # extract shapes
    N, C, H, W = x.shape
    _, _, H_prime, W_prime = dout.shape

    # initialize the gradient
    dx = np.zeros_like(x)

    # same as forward pass, iterate over the sampples
    for n in range(N): 
        # then over the channels
        for c in range(C): 
            # then over the output height
            for h in range(H_prime): 
                # then over the output width
                for w in range(W_prime): 
                    # defines the receptive field
                    h_start = h * stride
                    h_end = h_start + pool_height
                    
                    w_start = w * stride
                    w_end = w_start + pool_width

                    # extract the receptive field
                    x_field = x[n, c, h_start:h_end, w_start:w_end]

                    # find the index of the maximum value
                    max_index = np.unravel_index(np.argmax(x_field), x_field.shape)

                    # backpropagate the gradient

                    dx[n, c, h_start + max_index[0], w_start + max_index[1]] += dout[n, c, h, w]
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        sample_mean = np.mean(x, axis = 0)# calculate the sample mean
        sample_var = np.var(x, axis = 0) # calculate the sample variance

        x_norm = (x-sample_mean)/(np.sqrt(sample_var + eps)) # normalize the input
        out = gamma*x_norm + beta # scale and shift the normalized input

        # update the running mean and running variance values
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # necessary values for backprop calculations
        cache = (x, sample_mean, sample_var, x_norm, gamma, beta, eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        x_norm = (x-running_mean)/(np.sqrt(running_var + eps)) # normalize the input
        out = gamma*x_norm + beta # scale and shift the normalized input

        # necessary values for backprop calculations
        cache = (x, running_mean, running_var, x_norm, gamma, beta, eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################

    # retrieve forward propagation variables from cache
    (x, sample_mean, sample_var, x_norm, gamma, beta, eps) = cache

    # retreive num samples and num features
    N, D = x.shape

    # eq 1.1
    dbeta = np.sum(dout, axis = 0)

    # eq 1.2
    dgamma = np.sum(dout*x_norm, axis = 0)

    # eq 1.3
    dx_norm = gamma*dout

    # eq 1.4
    d1_4 = dx_norm*(1 / np.sqrt(sample_var + eps)) 

    # eq 1.5
    d1_5 = np.sum(dx_norm*(x - sample_mean), axis = 0)

    # eq 1.51
    d1_51 = -d1_5/(sample_var + eps)

    # eq 1.52
    d1_52 = 0.5 * 1 / (np.sqrt(sample_var + eps)) * d1_51

    # eq 1.53
    d1_53 = 1/N * np.ones((N,D)) * d1_52

    # eq 1.54
    d1_54 = 2*(x - sample_mean) * d1_53

    # eq 1.6
    dmu = -np.sum((d1_4 + d1_54), axis = 0)

    # eq 1.7
    d1_7 = d1_4 + d1_54

    # eq 1.8
    d1_8 = 1/N * np.ones((N,D))*dmu

    # eq 1.9
    dx = d1_7 + d1_8
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dgamma, dbeta