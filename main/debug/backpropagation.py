import numpy as np


def he_init(shape, n_input):
    """shape = ukuran weight untuk kernel
    n_input = jumlah input yang masuk ke satu neuron/output"""
    weight = np.random.standard_normal(shape)

    weight *= np.sqrt(2.0 / n_input)

    return weight.astype("float32")


def init_parameters_cnn():
    """

    Fully Connecteed dengan arsitektur 2 hidden layer

    """
    np.random.seed(1234)

    # Conv layer
    # Shape kernel yang umum: (jumlah_filter, channel_input, tinggi_kernel, lebar_kernel)
    W1 = he_init(shape=(8, 1, 3, 3), n_input=3 * 3 * 1)  # Untuk Kernel
    B1 = np.zeros(8, dtype="float32")

    # Dense layer 1
    # Setelah conv valid 3x3: 28 -> 26
    # Setelah maxpool 2x2: 26 -> 13
    # Flatten: 8 * 13 * 13 = 1352
    W2 = he_init(shape=(1352, 128), n_input=1352)
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


def softmax(x):
    """Softmax dengan stabilisasi numerik"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(Z3, y_true):
    """
    Hitung cross-entropy loss dari logits
    Z3: (N, num_classes) - logits
    y_true: (N,) - label integer
    """
    N = Z3.shape[0]
    probs = softmax(Z3)

    # Ambil probabilitas kelas yang benar
    correct_probs = probs[np.arange(N), y_true]

    # Cross-entropy: -log(p)
    loss = -np.mean(np.log(correct_probs + 1e-12))

    return loss


def check_gradients(x_batch, y_batch, params, epsilon=1e-5):
    """
    Verifikasi gradient dengan finite difference

    Args:
        x_batch: input batch (N, C, H, W)
        y_batch: label batch (N,)
        params: tuple (W1, b1, W2, b2, W3, b3)
        epsilon: kecilnya perubahan untuk finite difference

    Returns:
        dict berisi relative error untuk setiap parameter
    """
    W1, b1, W2, b2, W3, b3 = params

    # STEP 1: Hitung gradient ANALITIK dari backward_cnn
    Z3, layer_caches = forward_cnn(x_batch, params)
    loss = cross_entropy_loss(Z3, y_batch)

    # Hitung gradient analitik (Anda perlu implementasi backward_cnn)
    grads = backward_cnn(layer_caches, layer_caches, params)
    dW1, db1, dW2, db2, dW3, db3 = grads

    print(f"Loss awal: {loss:.6f}\n")

    # STEP 2: Hitung gradient NUMERIK dengan finite difference
    grad_num_W1 = np.zeros_like(W1)
    grad_num_b1 = np.zeros_like(b1)
    grad_num_W2 = np.zeros_like(W2)
    grad_num_b2 = np.zeros_like(b2)
    grad_num_W3 = np.zeros_like(W3)
    grad_num_b3 = np.zeros_like(b3)

    # Fungsi helper untuk hitung loss dengan parameter yang dimodifikasi
    def compute_loss_with_params(W1_mod, b1_mod, W2_mod, b2_mod, W3_mod, b3_mod):
        params_mod = (W1_mod, b1_mod, W2_mod, b2_mod, W3_mod, b3_mod)
        Z3_mod, _ = forward_cnn(x_batch, params_mod)
        return cross_entropy_loss(Z3_mod, y_batch)

    # Hitung gradient numerik untuk W1
    print("Menghitung gradient numerik W1...")
    it = np.nditer(W1, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index

        # Simpan nilai asli
        original_value = W1[idx]

        # f(x + epsilon)
        W1[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        # f(x - epsilon)
        W1[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        # Gradient numerik: (f(x+e) - f(x-e)) / (2*e)
        grad_num_W1[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        # Kembalikan nilai asli
        W1[idx] = original_value

        it.iternext()

    # Hitung gradient numerik untuk b1
    print("Menghitung gradient numerik b1...")
    it = np.nditer(b1, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = b1[idx]

        b1[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        b1[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        grad_num_b1[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        b1[idx] = original_value

        it.iternext()

    # Hitung gradient numerik untuk W2
    print("Menghitung gradient numerik W2...")
    it = np.nditer(W2, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = W2[idx]

        W2[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        W2[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        grad_num_W2[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        W2[idx] = original_value

        it.iternext()

    # Hitung gradient numerik untuk b2
    print("Menghitung gradient numerik b2...")
    it = np.nditer(b2, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = b2[idx]

        b2[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        b2[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        grad_num_b2[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        b2[idx] = original_value

        it.iternext()

    # Hitung gradient numerik untuk W3
    print("Menghitung gradient numerik W3...")
    it = np.nditer(W3, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = W3[idx]

        W3[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        W3[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        grad_num_W3[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        W3[idx] = original_value

        it.iternext()

    # Hitung gradient numerik untuk b3
    print("Menghitung gradient numerik b3...")
    it = np.nditer(b3, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = b3[idx]

        b3[idx] = original_value + epsilon
        loss_plus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        b3[idx] = original_value - epsilon
        loss_minus = compute_loss_with_params(W1, b1, W2, b2, W3, b3)

        grad_num_b3[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        b3[idx] = original_value

        it.iternext()

    # STEP 3: Bandingkan gradient analitik vs numerik
    print("\n" + "=" * 60)
    print("HASIL VERIFIKASI GRADIENT")
    print("=" * 60)

    def relative_error(grad_num, grad_ana):
        """Hitung relative error"""
        numerator = np.abs(grad_num - grad_ana)
        denominator = np.maximum(np.abs(grad_num) + np.abs(grad_ana), 1e-8)
        return numerator / denominator

    results = {
        "W1": relative_error(grad_num_W1, dW1),
        "b1": relative_error(grad_num_b1, db1),
        "W2": relative_error(grad_num_W2, dW2),
        "b2": relative_error(grad_num_b2, db2),
        "W3": relative_error(grad_num_W3, dW3),
        "b3": relative_error(grad_num_b3, db3),
    }

    for name, rel_err in results.items():
        max_err = np.max(rel_err)
        mean_err = np.mean(rel_err)

        print(f"\n{name}:")
        print(f"  Max error : {max_err:.2e}")
        print(f"  Mean error: {mean_err:.2e}")

        if max_err < 1e-7:
            print(f"  Status    : ✅ EXCELLENT (< 1e-7)")
        elif max_err < 1e-4:
            print(f"  Status    : ✅ GOOD (< 1e-4)")
        elif max_err < 1e-2:
            print(f"  Status    : ⚠️  ACCEPTABLE (< 1e-2)")
        else:
            print(f"  Status    : ❌ ERROR TOO LARGE!")

    return results


# Cara menggunakan:
if __name__ == "__main__":
    # Buat data dummy kecil untuk test (agar cepat)
    x_test = np.random.randn(2, 1, 28, 28).astype("float32")  # 2 gambar
    y_test = np.array([3, 7])  # label acak

    # Inisialisasi parameter
    params = init_parameters_cnn()

    # Jalankan gradient checking
    print("Memulai gradient checking...")
    print("(Ini akan memakan waktu karena finite difference lambat)")
    print("-" * 60)

    results = check_gradients(x_test, y_test, params, epsilon=1e-5)

    print("\n" + "=" * 60)
    print("INTERPRETASI HASIL:")
    print("=" * 60)
    print("✅ Jika semua error < 1e-4  → Backprop Anda BENAR!")
    print("⚠️  Jika error 1e-4 sampai 1e-2 → Mungkin ada bug kecil")
    print("❌ Jika error > 1e-2 → Ada bug serius di backprop")
