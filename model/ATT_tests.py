import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

# TensorFlow implementation
def tf_scaled_dot_product_attention(Q, K, V, dropout_rate=0., training=True):
    d_k = tf.cast(tf.shape(K)[-1], tf.float32)
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)
    outputs /= tf.sqrt(d_k)
    outputs = tf.nn.softmax(outputs)
    attention = tf.transpose(outputs, [0, 2, 1])
    if training and dropout_rate > 0:
        outputs = tf.nn.dropout(outputs, rate=dropout_rate)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
    return outputs, attention

# PyTorch implementation
class PyTorchScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_rate=0.):
        super(PyTorchScaledDotProductAttention, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        outputs = torch.matmul(attention, V)
        return outputs, attention.transpose(-2, -1)

# Test function
def test_scaled_dot_product_attention():
    # Set random seeds for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    if hasattr(tf, 'random'):
        tf.random.set_seed(0)
    else:
        tf.set_random_seed(0)

    # Define input shapes
    N, T_q, T_k, d_k, d_v = 2, 3, 4, 8, 8

    # Generate random inputs
    Q_np = np.random.randn(N, T_q, d_k).astype(np.float32)
    K_np = np.random.randn(N, T_k, d_k).astype(np.float32)
    V_np = np.random.randn(N, T_k, d_v).astype(np.float32)

    # TensorFlow
    Q_tf = tf.constant(Q_np)
    K_tf = tf.constant(K_np)
    V_tf = tf.constant(V_np)
    tf_output, tf_attention = tf_scaled_dot_product_attention(Q_tf, K_tf, V_tf, dropout_rate=0., training=False)
    
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        # TensorFlow 2.x
        tf_output, tf_attention = tf_output.numpy(), tf_attention.numpy()
    else:
        # TensorFlow 1.x
        with tf.Session() as sess:
            tf_output, tf_attention = sess.run([tf_output, tf_attention])

    # PyTorch
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)
    torch_attention = PyTorchScaledDotProductAttention(dropout_rate=0.)
    torch_attention.eval()  # Set to evaluation mode to disable dropout
    with torch.no_grad():
        torch_output, torch_attention = torch_attention(Q_torch, K_torch, V_torch)

    # Convert PyTorch outputs to numpy for comparison
    torch_output = torch_output.numpy()
    torch_attention = torch_attention.numpy()

    # Compare outputs
    output_diff = np.abs(tf_output - torch_output).max()
    attention_diff = np.abs(tf_attention - torch_attention).max()

    print(f"Maximum difference in output: {output_diff}")
    print(f"Maximum difference in attention: {attention_diff}")

    assert output_diff < 1e-5, "Outputs are not close enough"
    assert attention_diff < 1e-5, "Attention weights are not close enough"

    print("Test passed! TensorFlow and PyTorch implementations are equivalent.")

# Run the test
if __name__ == "__main__":
    test_scaled_dot_product_attention()


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        return torch.matmul(attention, V)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, queries, keys, values):
        Q = self.split_heads(self.W_Q(queries))
        K = self.split_heads(self.W_K(keys))
        V = self.split_heads(self.W_V(values))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_O(self.combine_heads(attn_output))
        
        # Add residual connection to the input queries
        return self.layer_norm(output + queries)

def test_multihead_attention():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define parameters
    batch_size, seq_length, d_model = 2, 5, 64
    num_heads = 8

    # Create an instance of MultiHeadAttention
    mha = MultiHeadAttention(d_model, num_heads)

    # Generate random input tensors
    Q = torch.randn(batch_size, seq_length, d_model)
    K = torch.randn(batch_size, seq_length, d_model)
    V = torch.randn(batch_size, seq_length, d_model)

    # Perform the multi-head attention operation
    output = mha(Q, K, V)

    # Check output shape
    assert output.shape == (batch_size, seq_length, d_model), f"Expected shape {(batch_size, seq_length, d_model)}, but got {output.shape}"

    # Check if output values are finite (not NaN or inf)
    assert torch.isfinite(output).all(), "Output contains NaN or inf values"

    print("All tests passed!")

if __name__ == "__main__":
    test_multihead_attention()

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seeds
np.random.seed(0)
tf.random.set_seed(0)
torch.manual_seed(0)

# TensorFlow implementation
def tf_scaled_dot_product_attention(Q, K, V, mask):
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights

class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(TFMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = tf_scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

# PyTorch implementation
class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(PyTorchMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = k.size()[-1]
        scaled_attention_logits = matmul_qk / np.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.dense(concat_attention)
        return output, attention_weights

# Test function
def test_multihead_attention_equivalence():
    # Parameters
    d_model = 512
    num_heads = 8
    batch_size = 4
    seq_length = 10

    # Create inputs
    np_input = np.random.randn(batch_size, seq_length, d_model).astype(np.float32)
    tf_input = tf.constant(np_input)
    torch_input = torch.from_numpy(np_input)

    # Create mask (assuming no mask for simplicity)
    mask = None

    # TensorFlow
    tf_mha = TFMultiHeadAttention(d_model, num_heads)
    tf_output, tf_attention = tf_mha(tf_input, tf_input, tf_input, mask)

    # PyTorch
    torch_mha = PyTorchMultiHeadAttention(d_model, num_heads)
    # Copy weights from TensorFlow to PyTorch
    torch_mha.wq.weight.data = torch.from_numpy(tf_mha.wq.weights[0].numpy().T)
    torch_mha.wq.bias.data = torch.from_numpy(tf_mha.wq.weights[1].numpy())
    torch_mha.wk.weight.data = torch.from_numpy(tf_mha.wk.weights[0].numpy().T)
    torch_mha.wk.bias.data = torch.from_numpy(tf_mha.wk.weights[1].numpy())
    torch_mha.wv.weight.data = torch.from_numpy(tf_mha.wv.weights[0].numpy().T)
    torch_mha.wv.bias.data = torch.from_numpy(tf_mha.wv.weights[1].numpy())
    torch_mha.dense.weight.data = torch.from_numpy(tf_mha.dense.weights[0].numpy().T)
    torch_mha.dense.bias.data = torch.from_numpy(tf_mha.dense.weights[1].numpy())

    torch_output, torch_attention = torch_mha(torch_input, torch_input, torch_input, mask)

    # Convert outputs to numpy for comparison
    tf_output_np = tf_output.numpy()
    torch_output_np = torch_output.detach().numpy()

    # Compare outputs
    max_diff = np.max(np.abs(tf_output_np - torch_output_np))
    print(f"Maximum difference in output: {max_diff}")

    assert np.allclose(tf_output_np, torch_output_np, atol=1e-5), "Outputs are not close enough"
    print("Test passed! TensorFlow and PyTorch implementations are equivalent.")

# Run the test
if __name__ == "__main__":
    test_multihead_attention_equivalence()
