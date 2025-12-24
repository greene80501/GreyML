import numpy as np

from greyml import Tensor
from greyml.nn.layers import Linear, LSTM, GRU, MultiheadAttention, Dropout


def test_linear_forward_shape():
    layer = Linear(4, 3)
    x = Tensor(np.ones((2, 4), dtype=np.float32))
    out = layer(x)
    assert out.shape == (2, 3)


def test_lstm_forward_shapes():
    lstm = LSTM(5, 8)
    x = Tensor(np.ones((3, 2, 5), dtype=np.float32))
    out, (h, c) = lstm.forward(x)
    assert out.shape == (3, 2, 8)
    assert h.shape == (2, 8)
    assert c.shape == (2, 8)


def test_gru_forward_shapes():
    gru = GRU(4, 6)
    x = Tensor(np.ones((2, 3, 4), dtype=np.float32))
    out, h = gru.forward(x)
    assert out.shape == (2, 3, 6)
    assert h.shape == (3, 6)


def test_multihead_attention_forward_shape():
    attn = MultiheadAttention(embed_dim=16, num_heads=4)
    batch, seq, embed = 2, 5, 16
    q = Tensor(np.random.randn(batch, seq, embed).astype(np.float32))
    k = Tensor(np.random.randn(batch, seq, embed).astype(np.float32))
    v = Tensor(np.random.randn(batch, seq, embed).astype(np.float32))
    out, weights = attn.forward(q, k, v)
    assert out.shape == (batch, seq, embed)
    assert weights.shape == (batch, attn.num_heads, seq, seq)


def test_dropout_respects_eval_mode():
    drop = Dropout(p=0.5)
    drop.training = False
    x = Tensor(np.ones((4, 4), dtype=np.float32))
    out = drop(x)
    np.testing.assert_allclose(out.numpy(), x.numpy())
