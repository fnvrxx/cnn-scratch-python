import numpy as np


def he_init(shape, n_input):
    """shape = ukuran weight untuk kernel
    n_input = jumlah input yang masuk ke satu neuron/output"""
    weight = np.random.standard_normal(shape)

    weight *= np.sqrt(2.0 / n_input)

    return weight.astype("float32")


def init_parameters_cnn():
    np.random.seed(42)

    # Conv layer
    # Shape kernel yang umum: (jumlah_filter, channel_input, tinggi_kernel, lebar_kernel)
    W1 = he_init(shape=(8, 1, 3, 3), n_input=3 * 3 * 1)  # Untuk Kernel
    B1 = np.zeros(8, dtype="float32")

    # Dense layer 1
    # Setelah conv valid 3x3: 28 -> 26
    # Setelah maxpool 2x2: 26 -> 13
    # Flatten: 8 * 13 * 13 = 1352
    # W2 = he_init(shape=(1352, 128), n_input=1352)
    # B2 = np.zeros(128, dtype="float32")  # 128 itu banyak neuron
    W2 = he_init(shape=(8, 128), n_input=8)  # ukuran dimensi 4x4 dengan
    B2 = np.zeros(128, dtype="float32")  # 128 itu banyak neuron

    # Output layer
    W3 = he_init(shape=(128, 10), n_input=128)
    B3 = np.zeros(10, dtype="float32")  # 10 itu banyak neuron

    return W1, B1, W2, B2, W3, B3


def im2col(x, kernel_size, stride=1):
    """
    Mengubah setiap patch gambar menjadi satu baris matrix.

    Input:
        x            : (N, C, H, W)
        kernel_size  : (KH, KW)
        stride       : langkah pergeseran kernel

    Output:
        cols         : (N * H_out * W_out, C * KH * KW)
    """

    N, C, H, W = x.shape

    KH, KW = kernel_size

    # Ukuran output convolution
    H_out = (H - KH) // stride + 1  # untuk menentukan ukuran kernel pada height
    W_out = (W - KW) // stride + 1  # untuk menentukan posisi  kernel pada weight

    # Tempat menyimpan semua patch
    cols = np.empty((N, H_out, W_out, C, KH, KW), dtype=x.dtype)

    # Ambil setiap posisi kernel
    for i in range(KH):
        for j in range(KW):

            cols[:, :, :, :, i, j] = x[
                :, :, i : i + stride * H_out : stride, j : j + stride * W_out : stride
            ].transpose(0, 2, 3, 1)

    # Ubah semua patch menjadi baris
    cols = cols.reshape(N * H_out * W_out, C * KH * KW)

    return cols


def conv2d_forward_ujicoba(x, W, b, stride=1, pad=0):

    n, c, h, width = x.shape  # N = image, C = Channel,h = height
    f, _, kh, kw = W.shape  # ambil nilai f, kh,kw (abaikan c dengan '_')

    h_out = (h + 2 * pad - kh) // stride + 1  # height for output/feature map
    w_out = (width + 2 * pad - kw) // stride + 1  # width for output/feature map
    x_pad = x

    cols = im2col(x_pad, kernel_size=(kh, kw), stride=stride)
    w_col = W.reshape(f, -1).T  # (C*KH*KW, F)

    z = cols @ w_col + b
    z = z.reshape(n, h_out, w_out, f).transpose(0, 3, 1, 2)

    cache = (x_pad, W, stride, pad, cols, x.shape)

    return z, cache


def maxpool_forward_ujicoba(x, size=2, stride=2):
    """
    Max pooling forward.
    """
    n, c, h, width = x.shape

    h_out = (h - size) // stride + 1
    w_out = (width - size) // stride + 1

    out = np.empty((n, c, h_out, w_out), dtype=x.dtype)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride

            window = x[:, :, h_start : h_start + size, w_start : w_start + size]

            out[:, :, i, j] = window.max(axis=(2, 3))

    cache = (x, size, stride)

    return out, cache


def flatten_forward(x):
    """
    Ubah feature map 4D menjadi 2D untuk dense layer.
    """
    return x.reshape(x.shape[0], -1), x.shape


def forward_cnn(x, params):
    W1, b1, W2, b2, W3, b3 = params

    # Conv 1
    Z1, cache_conv1 = conv2d_forward_ujicoba(x, W1, b1, stride=1, pad=0)
    A1 = np.maximum(Z1, 0)  # ReLU

    # Max pooling
    P1, cache_pool1 = maxpool_forward_ujicoba(A1, size=2, stride=2)

    # Flatten
    F, cache_flat = flatten_forward(P1)

    # Dense 1
    Z2 = F @ W2 + b2
    A2 = np.maximum(Z2, 0)  # ReLU

    # Output dense
    Z3 = A2 @ W3 + b3

    caches = {
        "conv1": cache_conv1,
        "relu1": Z1,
        "pool1": cache_pool1,
        "flat": cache_flat,
        "dense2": (F, W2),
        "relu2": Z2,
        "dense3": (A2, W3),
    }

    return Z3, caches


##################################
######## UJI COBA ################
##################################

# Input (asumsikan x adalah hasil matriks dari gambar)
# x = np.random.rand(1, 1, 28, 28).astype("float32")  # gambar berukuran 28 x 28
x = np.random.rand(1, 1, 4, 4).astype("float32")  # gambar berukuran 4 x 4

# inisialisasi bobot
params = init_parameters_cnn()

# Forward CNN
forward = forward_cnn(x, params)
print(forward)
