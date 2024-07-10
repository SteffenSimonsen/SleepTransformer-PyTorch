import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

# PyTorch implementation
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

# TensorFlow implementation
def ln_tensorflow(inputs, gamma, beta, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
    return gamma * normalized + beta

# Test function
def test_layer_norm_equivalence():
    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)

    # Create random input
    input_shape = (2, 3, 4)  # (batch_size, seq_len, features)
    x_np = np.random.randn(*input_shape).astype(np.float32)
    x_torch = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np)

    # Initialize parameters
    features = input_shape[-1]
    gamma_np = np.ones(features, dtype=np.float32)
    beta_np = np.zeros(features, dtype=np.float32)

    # PyTorch layer norm
    ln_torch = LayerNorm(features)
    ln_torch.gamma.data = torch.from_numpy(gamma_np)
    ln_torch.beta.data = torch.from_numpy(beta_np)

    # TensorFlow layer norm
    gamma_tf = tf.constant(gamma_np)
    beta_tf = tf.constant(beta_np)

    # Compute outputs
    output_torch = ln_torch(x_torch).detach().numpy()
    output_tf = ln_tensorflow(x_tf, gamma_tf, beta_tf).numpy()

    # Compare outputs
    np.testing.assert_allclose(output_tf, output_torch, rtol=1e-5, atol=1e-5)
    print("Test passed! The PyTorch and TensorFlow implementations are equivalent.")

# Run the test
test_layer_norm_equivalence()
