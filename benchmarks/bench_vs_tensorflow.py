"""
Simple CPU benchmarks comparing GreyML vs TensorFlow.

Runs a few NN/ML-style workloads:
- Dense matmul
- Elementwise add + ReLU
- Softmax + cross entropy
- MLP training step (forward + backward + SGD)
"""

import os
import sys
import time
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    print("TensorFlow import failed:", exc)
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import greyml  # noqa: E402
from greyml import Tensor  # noqa: E402
from greyml.tensor import cross_entropy  # noqa: E402
from greyml.tensor import _lib as _gm_lib  # noqa: E402


def timeit(fn, reps=10, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / reps


def report(name, tf_time, gm_time):
    ratio = (tf_time / gm_time) if gm_time > 0 else float("inf")
    print(f"{name}: tf {tf_time:.6f}s | greyml {gm_time:.6f}s | tf/greyml {ratio:.2f}x")


def bench_matmul():
    n = 512
    reps = 6
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    ga_a = Tensor(a)
    ga_b = Tensor(b)
    tf_a = tf.constant(a)
    tf_b = tf.constant(b)

    def run_tf():
        out = tf.matmul(tf_a, tf_b)
        _ = out.numpy()

    def run_gm():
        out = ga_a @ ga_b
        _ = out.numpy()

    tf_time = timeit(run_tf, reps=reps)
    gm_time = timeit(run_gm, reps=reps)
    report("matmul 512x512", tf_time, gm_time)


def bench_add_relu():
    reps = 8
    x = np.random.randn(1024, 1024).astype(np.float32)
    y = np.random.randn(1024, 1024).astype(np.float32)

    ga_x = Tensor(x)
    ga_y = Tensor(y)
    tf_x = tf.constant(x)
    tf_y = tf.constant(y)

    def run_tf():
        out = tf.nn.relu(tf_x + tf_y)
        _ = out.numpy()

    def run_gm():
        out = (ga_x + ga_y).relu()
        _ = out.numpy()

    tf_time = timeit(run_tf, reps=reps)
    gm_time = timeit(run_gm, reps=reps)
    report("add+relu 1024x1024", tf_time, gm_time)


def bench_softmax_ce():
    reps = 8
    batch = 256
    classes = 1000
    logits = np.random.randn(batch, classes).astype(np.float32)
    targets = np.random.randint(0, classes, size=(batch,), dtype=np.int64)

    ga_logits = Tensor(logits)
    ga_targets = Tensor(targets)
    tf_logits = tf.constant(logits)
    tf_targets = tf.constant(targets)
    tf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def run_tf():
        out = tf_loss(tf_targets, tf_logits)
        _ = out.numpy()

    def run_gm():
        out = cross_entropy(ga_logits, ga_targets, reduction="mean")
        _ = out.numpy()

    tf_time = timeit(run_tf, reps=reps)
    gm_time = timeit(run_gm, reps=reps)
    report("softmax+CE (256x1000)", tf_time, gm_time)


def bench_mlp_step():
    reps = 4
    batch = 128
    in_dim = 784
    hidden = 256
    classes = 10
    x = np.random.randn(batch, in_dim).astype(np.float32)
    y = np.random.randint(0, classes, size=(batch,), dtype=np.int64)

    ga_x = Tensor(x)
    ga_y = Tensor(y)
    l1 = greyml.nn.layers.Linear(in_dim, hidden)
    l2 = greyml.nn.layers.Linear(hidden, classes)
    opt = greyml.optim.SGD(l1.parameters() + l2.parameters(), lr=0.01)

    tf_x = tf.constant(x)
    tf_y = tf.constant(y)
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation="relu"),
        tf.keras.layers.Dense(classes),
    ])
    tf_opt = tf.keras.optimizers.SGD(0.01)
    tf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def run_tf():
        with tf.GradientTape() as tape:
            logits = tf_model(tf_x, training=True)
            loss = tf_loss(tf_y, logits)
        grads = tape.gradient(loss, tf_model.trainable_variables)
        tf_opt.apply_gradients(zip(grads, tf_model.trainable_variables))
        _ = loss.numpy()

    def run_gm():
        opt.zero_grad()
        h = l1(ga_x).relu()
        logits = l2(h)
        loss = cross_entropy(logits, ga_y, reduction="mean")
        loss.backward()
        opt.step()
        _ = loss.numpy()

    tf_time = timeit(run_tf, reps=reps, warmup=1)
    gm_time = timeit(run_gm, reps=reps, warmup=1)
    report("MLP step (128x784)", tf_time, gm_time)


def bench_conv2d():
    if not getattr(_gm_lib, "ga_conv2d", None):
        print("conv2d: skipped (C op not available)")
        return
    reps = 6
    batch = 4
    in_ch = 3
    out_ch = 8
    h = 32
    w = 32
    k = 3

    x_nchw = np.random.randn(batch, in_ch, h, w).astype(np.float32)
    w_oihw = np.random.randn(out_ch, in_ch, k, k).astype(np.float32)
    b = np.random.randn(out_ch).astype(np.float32)

    ga_x = Tensor(x_nchw)
    conv = greyml.nn.layers.Conv2d(in_ch, out_ch, k, stride=1, padding=1, bias=True)
    conv.weight = Tensor(w_oihw, requires_grad=True)
    conv.bias = Tensor(b, requires_grad=True)

    tf_x = tf.constant(np.transpose(x_nchw, (0, 2, 3, 1)))
    tf_w = tf.constant(np.transpose(w_oihw, (2, 3, 1, 0)))
    tf_b = tf.constant(b)

    def run_tf():
        out = tf.nn.conv2d(tf_x, tf_w, strides=1, padding="SAME")
        out = tf.nn.bias_add(out, tf_b)
        _ = out.numpy()

    def run_gm():
        out = conv(ga_x)
        _ = out.numpy()

    tf_time = timeit(run_tf, reps=reps)
    gm_time = timeit(run_gm, reps=reps)
    report("conv2d 4x3x32x32", tf_time, gm_time)


def bench_lstm_forward():
    reps = 4
    seq = 32
    batch = 32
    input_dim = 64
    hidden = 128

    x_seq = np.random.randn(seq, batch, input_dim).astype(np.float32)
    ga_x = Tensor(x_seq)
    lstm = greyml.nn.layers.LSTM(input_dim, hidden)

    tf_x = tf.constant(np.transpose(x_seq, (1, 0, 2)))
    tf_layer = tf.keras.layers.LSTM(hidden, return_sequences=True, return_state=True)

    def run_tf():
        out, h, c = tf_layer(tf_x, training=False)
        _ = out.numpy()
        _ = h.numpy()
        _ = c.numpy()

    def run_gm():
        out, (h, c) = lstm(ga_x)
        _ = out.numpy()
        _ = h.numpy()
        _ = c.numpy()

    tf_time = timeit(run_tf, reps=reps, warmup=1)
    gm_time = timeit(run_gm, reps=reps, warmup=1)
    report("LSTM forward (32x32x64)", tf_time, gm_time)


def main():
    np.random.seed(0)
    print("TensorFlow:", tf.__version__)
    print("GreyML:", greyml.__version__)
    bench_matmul()
    bench_add_relu()
    bench_softmax_ce()
    bench_mlp_step()
    bench_conv2d()
    bench_lstm_forward()


if __name__ == "__main__":
    main()
