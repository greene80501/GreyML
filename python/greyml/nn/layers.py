"""Neural network layers.
High-level Python layer definitions (Linear, Conv, RNN family) mapped onto the C backend.
"""

from typing import List
from .module import Module
from ..tensor import Tensor, _lib
import numpy as np


def _rand_weight(shape):
    return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=True)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _rand_weight((out_features, in_features))
        self.bias = _rand_weight((out_features,)) if bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.transpose()
        if self.bias is not None:
            output = output + self.bias
        return output


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = _rand_weight((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = _rand_weight((out_channels,)) if bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        if not _lib.ga_conv2d:
            # Fallback: return input (no-op) if binding missing
            return x
        bias_ptr = self.bias._c_ptr if self.bias is not None else None
        out_ptr = _lib.ga_conv2d(x._c_ptr, self.weight._c_ptr, bias_ptr,
                                 self.stride, self.padding, self.dilation, self.groups)
        return Tensor(_c_ptr=out_ptr, _shape=None, _dtype=np.float32)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        if not _lib.ga_max_pool2d:
            return x
        out_ptr = _lib.ga_max_pool2d(x._c_ptr, self.kernel_size, self.stride, self.padding, self.dilation)
        return Tensor(_c_ptr=out_ptr, _shape=None, _dtype=x.dtype)


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        if not _lib.ga_avg_pool2d:
            return x
        out_ptr = _lib.ga_avg_pool2d(x._c_ptr, self.kernel_size, self.stride, self.padding)
        return Tensor(_c_ptr=out_ptr, _shape=None, _dtype=x.dtype)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros((num_features,), dtype=np.float32)
        self.running_var = np.ones((num_features,), dtype=np.float32)
        self.gamma = _rand_weight((num_features,))
        self.beta = _rand_weight((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        # Simple batch stats on NCHW
        arr = x.numpy()
        mean = arr.mean(axis=(0, 2, 3))
        var = arr.var(axis=(0, 2, 3))
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        norm = (arr - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + self.eps)
        out = norm * self.gamma.numpy()[None, :, None, None] + self.beta.numpy()[None, :, None, None]
        return Tensor(out.astype(np.float32))


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return Tensor(x.numpy() * mask)


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = _rand_weight((num_embeddings, embedding_dim))

    def forward(self, x: Tensor) -> Tensor:
        idx = x.numpy().astype(np.int64)
        out = self.weight.numpy()[idx]
        return Tensor(out.astype(np.float32))


class SimpleRNN(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = _rand_weight((hidden_size, input_size))
        self.weight_hh = _rand_weight((hidden_size, hidden_size))
        self.bias_ih = _rand_weight((hidden_size,))
        self.bias_hh = _rand_weight((hidden_size,))

    def forward(self, x: Tensor, h0: Tensor = None):
        # x: (seq, batch, input)
        arr = x.numpy()
        seq, batch, _ = arr.shape
        h = h0.numpy() if h0 is not None else np.zeros((batch, self.hidden_size), dtype=np.float32)
        outputs = []
        for t in range(seq):
            xt = arr[t]
            h = np.tanh(
                xt @ self.weight_ih.numpy().T + self.bias_ih.numpy()
                + h @ self.weight_hh.numpy().T + self.bias_hh.numpy()
            )
            outputs.append(h.copy())
        out = np.stack(outputs, axis=0)
        return Tensor(out.astype(np.float32)), Tensor(h.astype(np.float32))


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = _rand_weight((4 * hidden_size, input_size))
        self.weight_hh = _rand_weight((4 * hidden_size, hidden_size))
        self.bias_ih = _rand_weight((4 * hidden_size,))
        self.bias_hh = _rand_weight((4 * hidden_size,))

    def forward(self, x: Tensor, h0_c0=None):
        arr = x.numpy()
        seq, batch, _ = arr.shape
        if h0_c0 is not None:
            h0, c0 = h0_c0
            h = h0.numpy()
            c = c0.numpy()
        else:
            h = np.zeros((batch, self.hidden_size), dtype=np.float32)
            c = np.zeros((batch, self.hidden_size), dtype=np.float32)

        wih = self.weight_ih.numpy()
        whh = self.weight_hh.numpy()
        bih = self.bias_ih.numpy()
        bhh = self.bias_hh.numpy()
        hid = self.hidden_size

        outputs = []
        for t in range(seq):
            xt = arr[t]
            gi = xt @ wih[:hid].T + bih[:hid] + h @ whh[:hid].T + bhh[:hid]
            gf = xt @ wih[hid:2*hid].T + bih[hid:2*hid] + h @ whh[hid:2*hid].T + bhh[hid:2*hid]
            gg = xt @ wih[2*hid:3*hid].T + bih[2*hid:3*hid] + h @ whh[2*hid:3*hid].T + bhh[2*hid:3*hid]
            go = xt @ wih[3*hid:].T + bih[3*hid:] + h @ whh[3*hid:].T + bhh[3*hid:]

            i_gate = 1.0 / (1.0 + np.exp(-gi))
            f_gate = 1.0 / (1.0 + np.exp(-gf))
            g_gate = np.tanh(gg)
            o_gate = 1.0 / (1.0 + np.exp(-go))

            c = f_gate * c + i_gate * g_gate
            h = o_gate * np.tanh(c)
            outputs.append(h.copy())

        out = np.stack(outputs, axis=0)
        return Tensor(out.astype(np.float32)), (Tensor(h.astype(np.float32)), Tensor(c.astype(np.float32)))


class GRU(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = _rand_weight((3 * hidden_size, input_size))
        self.weight_hh = _rand_weight((3 * hidden_size, hidden_size))
        self.bias_ih = _rand_weight((3 * hidden_size,))
        self.bias_hh = _rand_weight((3 * hidden_size,))

    def forward(self, x: Tensor, h0: Tensor = None):
        arr = x.numpy()
        seq, batch, _ = arr.shape
        h = h0.numpy() if h0 is not None else np.zeros((batch, self.hidden_size), dtype=np.float32)

        wih = self.weight_ih.numpy()
        whh = self.weight_hh.numpy()
        bih = self.bias_ih.numpy()
        bhh = self.bias_hh.numpy()
        hid = self.hidden_size

        outputs = []
        for t in range(seq):
            xt = arr[t]
            gr = xt @ wih[:hid].T + bih[:hid] + h @ whh[:hid].T + bhh[:hid]
            gz = xt @ wih[hid:2*hid].T + bih[hid:2*hid] + h @ whh[hid:2*hid].T + bhh[hid:2*hid]

            r_gate = 1.0 / (1.0 + np.exp(-gr))
            z_gate = 1.0 / (1.0 + np.exp(-gz))

            gn = xt @ wih[2*hid:].T + bih[2*hid:] + (r_gate * h) @ whh[2*hid:].T + bhh[2*hid:]
            n_gate = np.tanh(gn)

            h = (1.0 - z_gate) * n_gate + z_gate * h
            outputs.append(h.copy())

        out = np.stack(outputs, axis=0)
        return Tensor(out.astype(np.float32)), Tensor(h.astype(np.float32))


class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.wq = _rand_weight((embed_dim, embed_dim))
        self.wk = _rand_weight((embed_dim, embed_dim))
        self.wv = _rand_weight((embed_dim, embed_dim))
        self.wo = _rand_weight((embed_dim, embed_dim))

    def _split_heads(self, x):
        # x: (batch, seq, embed)
        b, s, _ = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))  # (b, h, s, d)

    def forward(self, query: Tensor, key: Tensor, value: Tensor):
        q = query.numpy()
        k = key.numpy()
        v = value.numpy()
        b, sq, _ = q.shape
        _, sk, _ = k.shape
        q_proj = q @ self.wq.numpy().T
        k_proj = k @ self.wk.numpy().T
        v_proj = v @ self.wv.numpy().T
        qh = self._split_heads(q_proj)
        kh = self._split_heads(k_proj)
        vh = self._split_heads(v_proj)
        scores = np.matmul(qh, np.transpose(kh, (0, 1, 3, 2))) / np.sqrt(self.head_dim)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        ctx = np.matmul(attn, vh)  # (b, h, sq, d)
        ctx = np.transpose(ctx, (0, 2, 1, 3)).reshape(b, sq, self.embed_dim)
        out = ctx @ self.wo.numpy().T
        return Tensor(out.astype(np.float32)), Tensor(attn.astype(np.float32))
