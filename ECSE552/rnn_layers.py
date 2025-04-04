import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  # find the next hidden state
  next_h = np.tanh(np.dot(x, Wx) + np.dot(prev_h, Wh) + b)

  # store the values for backprop
  cache = (x, prev_h, Wx, Wh, b, next_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  # retreive from cache
  x, prev_h, Wx, Wh, b, next_h = cache

  # calculate the gradient of the tanh activation function
  da = dnext_h * (1 - next_h**2) # eq 1

  # calculate the gradients
  dx = np.dot(da, Wx.T) # w.r.t x, eq 2
  dWx = np.dot(x.T, da) # w.r.t input-to-hidden weights, eq 3

  dprev_h = np.dot(da, Wh.T) # w.r.t prev hidden state, eq 4
  dWh = np.dot(prev_h.T, da) # hidden-to-hidden weights, eq 5

  db = np.sum(da, axis = 0) # w.r.t bias, eq 6
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape  # Extract input dimensions
  H = h0.shape[1]  # Extract hidden state dimensions

  h = np.zeros((N, T, H))  # Initialize hidden states storage
  cache = []  # Store values needed for backpropagation

  prev_h = h0  # Set initial hidden state

  for t in range(T):
    xt = x[:, t, :]  # Extract input for current timestep
    next_h, cache_t = rnn_step_forward(xt, prev_h, Wx, Wh, b)  # Forward step
    h[:, t, :] = next_h  # Store hidden state
    cache.append(cache_t)  # Store cache for backprop
    prev_h = next_h  # Update previous hidden state

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  N, T, H = dh.shape  # Extract dimensions
  D = cache[0][0].shape[1]  # Extract input dimension from cache tuple

  dx = np.zeros((N, T, D))  # Initialize gradient of inputs
  dWx = np.zeros((D, H))  # Initialize gradient of input-to-hidden weights
  dWh = np.zeros((H, H))  # Initialize gradient of hidden-to-hidden weights
  db = np.zeros(H)  # Initialize gradient of biases
  dprev_h = np.zeros((N, H))  # Initialize gradient of previous hidden state

  # Iterate over the reverse of the time steps
  for t in reversed(range(T)):
      dht = dh[:, t, :] + dprev_h  # Add upstream gradient to current timestep

      cache_t = cache[t]  # Retrieve cache for current timestep

      dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dht, cache_t)  # Backward step

      dx[:, t, :] = dx_t  # Store gradient of inputs
      dWx += dWx_t  # Accumulate gradient of input-to-hidden weights
      dWh += dWh_t  # Accumulate gradient of hidden-to-hidden weights
      db += db_t  # Accumulate gradient of biases

  dh0 = dprev_h  # Gradient of initial hidden state

  return dx, dh0, dWx, dWh, db

# not sure why this isn't in the original code, but it is used in the LSTM step forward function
def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))

def lstm_step_forward(x_t, h_prev, c_prev, W_x, W_h, b):
    """
    Forward pass for a single timestep of an LSTM.

    Inputs:
    - x_t: Input data at current timestep, shape (N, D)
    - h_prev: Previous hidden state, shape (N, H)
    - c_prev: Previous cell state, shape (N, H)
    - W_x: Input-to-hidden weight matrix, shape (D, 4H)
    - W_h: Hidden-to-hidden weight matrix, shape (H, 4H)
    - b: Bias vector, shape (4H,)

    Returns:
    - h_t: Next hidden state, shape (N, H)
    - c_t: Next cell state, shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """

    H = h_prev.shape[1]  # Hidden state dimension

    # Compute activation vector
    a = np.dot(x_t, W_x) + np.dot(h_prev, W_h) + b  

    # Split activation into four components
    a_i, a_f, a_o, a_g = np.split(a, 4, axis=1)  

    # Compute gate activations
    i_t = sigmoid(a_i)  # Input gate
    f_t = sigmoid(a_f)  # Forget gate
    o_t = sigmoid(a_o)  # Output gate
    g_t = np.tanh(a_g)  # Block input

    # Compute next cell state
    c_t = f_t * c_prev + i_t * g_t  

    # Compute next hidden state
    h_t = o_t * np.tanh(c_t)  

    # Store cache for backward pass
    cache = (x_t, h_prev, c_prev, W_x, W_h, b, i_t, f_t, o_t, g_t, c_t, h_t)

    return h_t, c_t, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  # retrieve from cache
  (x_t, h_prev, c_prev, W_x, W_h, b, i_t, f_t, o_t, g_t, c_t, h_t) = cache

  # derivates of the next hidden state through the tanh and output gate
  do_t = dnext_h * np.tanh(c_t) # eq 1
  dc_t = dnext_h * o_t * (1 - np.tanh(c_t)**2) + dnext_c # eq 2

  # gradient of the gates
  di_t = dc_t * g_t # eq 3
  dg_t = dc_t * i_t # eq 4
  df_t = dc_t * c_prev # eq 5
  dprev_c = dc_t * f_t # eq 6

  # derivatives of activation functions
  da_i = di_t * i_t * (1 - i_t) # eq 7
  da_f = df_t * f_t * (1 - f_t) # eq 8
  da_o = do_t * o_t * (1 - o_t) # eq 9
  da_g = dg_t * (1 - g_t**2) # eq 10

  # concatenate the gradients of the activation functions
  da = np.hstack((da_i, da_f, da_o, da_g))

  # gradients w.r.t. the input data, weights and biases
  dx = np.dot(da, W_x.T) # eq 11
  dWx = np.dot(x_t.T, da) # eq 12
  dWh = np.dot(h_prev.T, da) # eq 13
  dprev_h = np.dot(da, W_h.T) # eq 14
  db = np.sum(da, axis = 0) # eq 15
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape  # Extract input dimensions
  H = h0.shape[1]  # Extract hidden state dimensions

  h = np.zeros((N, T, H))  # Initialize hidden states storage
  c_prev = np.zeros((N, H))  # Initialize cell state storage
  h_prev = h0  # Set initial hidden state
  cache = []  # Store values needed for backpropagation

  # loop over the time steps
  for t in range(T):
    xt = x[:, t, :]  # Extract input for current timestep
    h_t, c_prev, cache_t = lstm_step_forward(xt, h_prev, c_prev, Wx, Wh, b)  # Forward step
    h[:, t, :] = h_t  # Store hidden state
    cache.append(cache_t)  # Store cache for backprop
    h_prev = h_t  # Update previous hidden state
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  pass
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db
