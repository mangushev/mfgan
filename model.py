
#TODO:
#attention masks - encoder, mask future "next items"
#how many layers in discriminator MLP
#layer normalization - replace keras, can new tfa
#review tf.gather

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import math
import six

import numpy as np

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask

def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)

def conv1d_layer(inputs, filter_width, in_channels, out_channels, padding, activation, initializer, trainable=True, name="conv"):
  with tf.compat.v1.variable_scope(name):
    filter = tf.compat.v1.get_variable(initializer=initializer, shape=[filter_width, in_channels, out_channels], trainable=trainable, name='filter')
    conv = tf.nn.conv1d(inputs, filter, [1], padding=padding, name="conv")
    bias = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[out_channels], trainable=trainable, name='bias')
    conv_bias = tf.nn.bias_add(conv, bias, name='conv_bias')
    conv_bias_relu = activation(conv_bias, name='conv_bias_relu')
    return conv_bias_relu

def print_shape(tensor, rank, tensor_name):
  #return tensor
  tensor_shape = get_shape_list(tensor, expected_rank=rank)
  return tf.Print(tensor, [tensor_shape], tensor_name, summarize=8)

def dense_layer(input_tensor, hidden_size, activation, initializer, name="dense"):
  with tf.compat.v1.variable_scope(name):
    input_shape = get_shape_list(input_tensor)

    if len(input_shape) != 2 and len(input_shape) != 3:
      assert_rank(tensor, expected_rank, tensor.name)

    batch_size = input_shape[0]
    if len(input_shape) == 3:
      seq_length = input_shape[1]
      input_width = input_shape[2]
      x = tf.reshape(input_tensor, [-1, input_width])
    else:
      input_width = input_shape[1]
      x = input_tensor

    w = tf.compat.v1.get_variable(initializer=initializer, shape=[input_width, hidden_size], name="w")
    z = tf.matmul(x, w, transpose_b=False)
    b = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[hidden_size], name="b")
    y = tf.nn.bias_add(z, b)
    if (activation):
      y = activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [batch_size, seq_length, hidden_size])

    return y

def layer_norm(input_tensor, trainable=True, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  #return tf.keras.layers.LayerNormalization(name=name,trainable=trainable,axis=-1,epsilon=1e-14,dtype=tf.float32)(input_tensor)
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, trainable=trainable, scope=name)

def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=768,
                      intermediate_act_fn=tf.nn.relu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def Discriminator(input_tensor, num_hidden_layers, num_items, hidden_size, batch_size, max_seq_length, num_attention_heads, activation_fn, initializer_range, factor_id, dropout_prob=0.2):

  #1). discriminator embedding table: [I, d] so like this [num_items, hidden size]
  #lookup in embeddings table to find entry for item
  with tf.compat.v1.variable_scope("input_embeddings"):
    #[I, d]
    embedding_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
          shape=[num_items, hidden_size], name='embedding_table')
 
    #[I, d] --> [0, I, d]
    embedding_expanded = tf.expand_dims(embedding_table, 0)
    #[0, I, d] --> [B, I, d]
    embedding_expanded = tf.tile(embedding_expanded, [batch_size, 1, 1])
  
    #let's do one_hot to dense for now to gather items
    #[B, n+1, I] --> [B, n+1] where n becomes an index
    input_dense_tensor = tf.argmax(input_tensor, axis=2)

    #[B, I, d], [B, n] --> [B, n, d]
    embedding = tf.gather(embedding_expanded, input_dense_tensor, axis=1, batch_dims=1, name="embedding")

    #self._embedding = tf.Print(_embedding, [self._embedding_table[0], self._embedding_table[1], self._embedding_table[2], self._embedding_table[3], self._embedding_table[4], self._embedding_table[5], self._embedding_table[6], _embedding], "_embedding", summarize=8)
  
  #2) make positional encoding
  #[n, d]
  with tf.compat.v1.variable_scope("input_positions"):
    position_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
          shape=[max_seq_length, hidden_size], name='position_table')
  
    #[B, n, d] + [0, n, d] --> [B n, d]
    embedding_with_positions = embedding + tf.expand_dims(position_table, 0)

    #self._generator_embedding_with_positions = tf.Print(_generator_embedding_with_positions, [_generator_embedding_with_positions], "_generator_embedding_with_positions", summarize=8)
  
  #3). discriminator transformer
    with tf.compat.v1.variable_scope("discriminator_transformer"):
      discriminator_tensor = transformer_model(embedding_with_positions,
                        attention_mask=None,
                        hidden_size=hidden_size,
                        num_hidden_layers=num_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_size,
                        intermediate_act_fn=activation_fn,
                        hidden_dropout_prob=dropout_prob,
                        attention_probs_dropout_prob=dropout_prob,
                        initializer_range=0.02,
                        do_return_all_layers=False)
  #4) MLP

#    #4). three fully connected layers
#    #[A, k1] --> [A, k2]
#    with tf.variable_scope("layer_3fc"):
#      layer_output = final_gru_output
#      for i in range(3):
#        with tf.variable_scope("fc_%d" %i):
#          layer_output = dense_layer(layer_output, k2, activation=None, initializer=create_initializer(initializer_range))
#          #layer_output = layer_norm(layer_output)
#          layer_output = tf.nn.relu(layer_output)
#          layer_output = dropout(layer_output, dropout_prob)

#    #[A, k2] --> [A, m]
#    self._next_feature= dense_layer(layer_output, num_features, activation=None, initializer=create_initializer(initializer_range))

  #[B, d] --> [B, 1]
  factor = dense_layer(discriminator_tensor[:,-2,:], 1, activation=tf.math.sigmoid, initializer=create_initializer(initializer_range), name="factor")

  #[B, 1] --> [B]
  return tf.squeeze(factor, axis=-1, name='factor_squeeze')

def calculate_rewards(factors, factor_bin_sizes, discriminator_hidden_layers, num_items, hidden_size, batch_size, max_seq_length, num_attention_heads, activation_fn, initializer_range, name='rewards', dropout_prob=0.2):
  with tf.compat.v1.variable_scope(name):
    #[B, f, n]
    factors_shape = get_shape_list(factors, expected_rank=3)
    num_factors = factors_shape[1]

    rewards = tf.TensorArray(size=num_factors, dtype=tf.float32, name="factors")
    #[B, n, I]
    for i in range(num_factors):
      with tf.variable_scope("unit_index_%d" %i):
        #[B, f, n] --> [B, n] --> [B, n, f_len]
        factor = tf.one_hot(factors[:,i,:], factor_bin_sizes[i], axis=-1)
        rewards = rewards.write(i, Discriminator(factor, discriminator_hidden_layers, num_items, hidden_size, batch_size, max_seq_length, num_attention_heads, activation_fn, initializer_range, i, dropout_prob=dropout_prob))

  #[B] x factors --> [f, B] --> [B, f] --> [B]
  return tf.math.reduce_mean(tf.transpose(rewards.stack(), [1,0]), axis=-1, keepdims=False)

class MfGAN(object):
  #   B = batch size (number of sequences)
  #   n = `from_tensor` sequence of historical interactions
  #   T = `to_tensor` sequence length in frames
  #   I - number of items
  #   d - d-dimentional dense representation/embeddings, hidden size
  #   f - number of factors
  def __init__(self,
               input_dense,
               factors,
               products,
               num_items,
               factor_bin_sizes,
               predictive_position=2,
               hidden_size=50,
               generator_hidden_layers=2,
               discriminator_hidden_layers=1,
               num_attention_heads=2,
               initializer_range=0.02,
               activation_fn=tf.nn.relu,
               dropout_prob=0.2,
               is_training=True):
    #dense input
    #[B, n]
    input_shape = get_shape_list(input_dense, expected_rank=2)
    batch_size = input_shape[0]
    max_seq_length = input_shape[1]

    if is_training == False:
       dropout_prob = 0.0

    #add item ids as factors
    #[B, f-1, n] --> [B, f, n]
    factors = tf.concat([tf.expand_dims(input_dense, 1), factors], axis=1)

    #and adjust bin sizes
    #[B, f-1] --> [B, f]
    factor_bin_sizes = tf.concat([[num_items], factor_bin_sizes], axis=0)

    #input as onehot
    #[B, n, I]
    input_tensor = tf.one_hot(input_dense, num_items, axis=-1)

    #[B, n]
    input_masks = tf.concat([tf.ones([batch_size, max_seq_length-predictive_position], tf.int32), tf.zeros([batch_size, predictive_position], tf.int32)], axis=1)    
    #input_t = tf.Print(input_tensor, [input_tensor], "input_tensor", summarize=100)

    #1). generator embedding table: [I, d] so like this [number_of_items, hidden_size]
    #lookup in embeddings table to find entry for item
    with tf.compat.v1.variable_scope("input_embeddings"):
      #[I, d]
      self._generator_embedding_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[num_items, hidden_size], name='generator_embedding_table')
  
      #[I, d] --> [0, I, d]
      generator_embedding_expanded = tf.expand_dims(self._generator_embedding_table, 0)
      #[0, I, d] --> [B, I, d]
      generator_embedding_expanded = tf.tile(generator_embedding_expanded, [batch_size, 1, 1])
  
      #let's do one_hot to dense for now to gather items
      #[B, n, I] --> [B, n] where n becomes an index
      #input_dense = tf.argmax(input_tensor, axis=2)

      #[B, I, d] --> [B, n, d]
      self._generator_embedding = tf.gather(generator_embedding_expanded, input_dense, axis=1, batch_dims=1, name="generator_embedding")

    #self._generator_embedding = tf.Print(_generator_embedding, [self._generator_embedding_table[0], self._generator_embedding_table[1], self._generator_embedding_table[2], self._generator_embedding_table[3], self._generator_embedding_table[4], self._generator_embedding_table[5], self._generator_embedding_table[6], _generator_embedding], "_generator_embedding", summarize=8)
  
    #2) make positional encoding
    #[n, d]
    with tf.compat.v1.variable_scope("input_positions"):
      self._encoder_position_table = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
            shape=[max_seq_length, hidden_size], name='encoder_position_table')
  
      #[B, n, d] + [0, n, d] --> [B, n, d]
      self._generator_embedding_with_positions = self._generator_embedding + tf.expand_dims(self._encoder_position_table, 0)

    #self._generator_embedding_with_positions = tf.Print(_generator_embedding_with_positions, [_generator_embedding_with_positions], "_generator_embedding_with_positions", summarize=8)
  
    #3) create 3D mask from 2D to mask attention with shorter sentenses then max_seq_length sentence
    attention_mask = create_attention_mask_from_input_mask(
              input_tensor, input_masks)
  
    #4). generator block
    with tf.compat.v1.variable_scope("generator_block"):
      self._generator_tensor = transformer_model(self._generator_embedding_with_positions,
                        attention_mask=attention_mask,
                        hidden_size=hidden_size,
                        num_hidden_layers=generator_hidden_layers,
                        num_attention_heads=num_attention_heads,
                        intermediate_size=hidden_size,
                        intermediate_act_fn=activation_fn,
                        hidden_dropout_prob=dropout_prob,
                        attention_probs_dropout_prob=dropout_prob,
                        initializer_range=0.02,
                        do_return_all_layers=False)
  
    #5) Predection Layer
    with tf.compat.v1.variable_scope("prediction_layer"):
      #[B, n, d] --> [B, d]
      new_item  = self._generator_tensor[:,-predictive_position,:]
      #[B, d] . [I, d]T --> [B, I] - this is onehot for next item
      next_item = tf.nn.softmax(tf.matmul(new_item, self._generator_embedding_table, transpose_b=True, name='next_item'))

      #[B, n, I] --> [B, n, I]
      self._generated_sample = tf.cond(tf.math.equal(predictive_position, 2), 
        lambda: tf.concat([input_tensor[:,:-2,:], tf.expand_dims(next_item, axis=1), input_tensor[:,-1:,:]], axis=1),
        lambda: tf.concat([input_tensor[:,:-1,:], tf.expand_dims(next_item, axis=1)], axis=1))

    #6) Factors loop and Q function, lambda = 0 --> taking mean of factors
    with tf.compat.v1.variable_scope("discriminator"):
      #[B, I] --> [B, 1]
      next_item_dense = tf.expand_dims(tf.argmax(next_item, axis=1), axis=-1)

      #[B, I, f], [B, 1] --> [B, 1, f]
      next_factors = tf.gather(products, next_item_dense, axis=1, batch_dims=1, name="next_factors")

      #[B, 1] + [B, 1, f-1] --> [B, 1, 1] + [B, 1, f-1] --> [B, 1, f]
      next_factors = tf.concat([tf.cast(tf.expand_dims(next_item_dense, axis=1), dtype=tf.int32), next_factors], axis=-1)      

      #factors are not used for prediction or testing and, for training, it is always the one before the last
      #[B, f, n] --> [B, f, n]
      generated_factors = tf.concat([factors[:,:,:-2], tf.transpose(next_factors, [0,2,1]), factors[:,:,-1:]], axis=-1)
     
      #Discriminator value from input
      #[B]
      self._input_item_Q = calculate_rewards(factors, factor_bin_sizes, discriminator_hidden_layers, num_items, hidden_size, batch_size, max_seq_length, num_attention_heads, activation_fn, initializer_range, name="based_on_input", dropout_prob=dropout_prob)

      #Discriminator value from generator
      #[B]
      self._generated_item_Q = calculate_rewards(generated_factors, factor_bin_sizes, discriminator_hidden_layers, num_items, hidden_size, batch_size, max_seq_length, num_attention_heads, activation_fn, initializer_range, name="from_generator", dropout_prob=dropout_prob)

    #7) log G to calculate gradients
    #[B, I]
    # mask all except one by multiplying on onehot label
    self._log_G = tf.math.log(next_item)*input_tensor[:,-2,:]

  @property
  def generated_sample(self):
    return tf.argmax(self._generated_sample, axis=2)

  @property
  def Input_item_Q(self):
    return self._input_item_Q

  @property
  def Generated_item_Q(self):
    return self._generated_item_Q

  @property
  def log_G(self):
    return self._log_G
