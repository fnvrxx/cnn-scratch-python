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
