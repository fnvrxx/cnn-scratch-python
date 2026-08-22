import numpy as np


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

    N, C, H, W_input = x.shape
    F, _, KH, KW = W.shape

    x_pad = x

    # Ambil semua patch
    cols = im2col(x_pad, kernel_size=(KH, KW), stride=stride)

    # Ubah semua kernel menjadi matrix
    W_col = W.reshape(F, -1).T

    # Convolution = matrix multiplication
    Z = cols @ W_col + b

    # Ukuran output
    H_out = (H + 2 * pad - KH) // stride + 1
    W_out = (W_input + 2 * pad - KW) // stride + 1

    # Kembalikan ke bentuk feature map
    Z = Z.reshape(N, H_out, W_out, F)

    Z = Z.transpose(0, 3, 1, 2)

    return Z


x = np.array(
    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]], dtype="float32"
)


# N = GAMBAR -> 1
# C = CHANNEL/WARNA -> 1
# H = HEIGHT -> 4
# W = WEIGHT -> 4
# ARRAY x yang diatas itu merupakan hasil visualnya

x = x.reshape(1, 1, 4, 4)  # N=1, C=1, H=4, W=4
print("x: ", x)
print("shape x:", x.shape)

# W -> weight untuk kernel
# W = np.array([
#     [
#         [1, 0],
#         [0, 1]
#     ]
# ], dtype='float32')
W = np.array([[[1, 0], [0, 1], [1, 1], [1, 1]]], dtype="float32")


# W = W.reshape(1, 1, 2, 2) #eksperimen pertama
W = W.reshape(2, 1, 2, 2)

b = np.array([0], dtype="float32")

cols = im2col(x, kernel_size=(2, 2), stride=1)

print("cols:")
print(cols)
print("shape:", cols.shape)

Z = conv2d_forward_ujicoba(x, W, b, stride=1, pad=0)
print("w:", W.shape)
print(Z)
print(
    Z.shape
)  # output shape ini seperti N,F,H,W (IMAGE : 1, 2 FEATURE MAP, 3 PIXEL UNTUK HEIGHT, 3 PIXEL UNTUK WEIGHT)

# Uji coba reshape
x = np.array(
    [[[1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]]], dtype="float32"
)  # total shape 12

# x = x.reshape(1, 1, 4, 3) #
# print("x: ",x)
