def conv2d_backward(dz, cache):
    """
    Backward convolution.
    """
    x_pad, w, stride, cols, x_shape = cache
    n, c, h, width = x_shape
    f, _, kh, kw = w.shape

    h_out, w_out = dz.shape[2], dz.shape[3]

    w_col = w.reshape(f, -1).T

    dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(n * h_out * w_out, f)

    # Gradient untuk weight dan bias
    dw_col = cols.T @ dz_reshaped
    dw = dw_col.T.reshape(w.shape)
    db = dz_reshaped.sum(axis=0)

    # Gradient untuk input
    dcols = dz_reshaped @ w_col.T
    dcols = dcols.reshape(n, h_out, w_out, c, kh, kw)

    dx_pad = np.zeros_like(x_pad)

    for i in range(kh):
        for j in range(kw):
            patch = dcols[:, :, :, :, i, j]
            patch = patch.transpose(0, 3, 1, 2)  # (N, C, H_out, W_out)

            dx_pad[
                :, :, i : i + stride * h_out : stride, j : j + stride * w_out : stride
            ] += patch

    dx = dx_pad

    return dx, dw, db


def maxpool_backward(dout, cache):
    """
    Max pooling backward.
    """
    x, size, stride = cache
    n, c, h, width = x.shape

    h_out, w_out = dout.shape[2], dout.shape[3]

    dx = np.zeros_like(x)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            w_start = j * stride

            window = x[:, :, h_start : h_start + size, w_start : w_start + size]

            window_max = window.max(axis=(2, 3), keepdims=True)
            mask = window == window_max

            dx[:, :, h_start : h_start + size, w_start : w_start + size] += (
                mask * dout[:, :, i, j][:, :, None, None]
            )

    return dx


def flatten_backward(dout, shape):
    """
    Kembalikan shape dari flatten.
    """
    return dout.reshape(shape)


def backward_cnn(dz3, caches, params):
    W1, b1, W2, b2, W3, b3 = params

    # Backward dense output
    A2, W3 = caches["dense3"]
    dW3 = A2.T @ dz3
    db3 = dz3.sum(axis=0)
    dA2 = dz3 @ W3.T

    # Backward ReLU dense 1
    dz2 = dA2 * (caches["relu2"] > 0)

    # Backward dense 1
    F, W2 = caches["dense2"]
    dW2 = F.T @ dz2
    db2 = dz2.sum(axis=0)
    dF = dz2 @ W2.T

    # Backward flatten
    dP1 = flatten_backward(dF, caches["flat"])

    # Backward maxpool
    dA1 = maxpool_backward(dP1, caches["pool1"])

    # Backward ReLU conv
    dz1 = dA1 * (caches["relu1"] > 0)

    # Backward conv
    dX, dW1, db1 = conv2d_backward(dz1, caches["conv1"])

    return dW1, db1, dW2, db2, dW3, db3
