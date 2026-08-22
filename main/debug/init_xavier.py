# xavier/glorot
def init_weight(shape, n_input, n_output):
    limit = np.sqrt(6 / (n_input + n_output))
    return np.random.uniform(-limit, limit, size=shape)


def init_parameter():
    # Input -> W1
    W1 = init_weight(
        shape=(8, 3, 3, 1),
        fan_in=3 * 3 * 1,  # 3 x 3 x1,
        fan_out=3 * 3 * 8,  # 3 * 3 * 8
    ).astype("float32")
    # B1
    B1 = np.zeros(8, dtype="float32")
    # W2
    W2 = init_weight(shape=(1352, 128), fan_in=1352, fan_out=128).astype(  # ?,  # ?
        "float32"
    )
    # B2
    B2 = np.zeros(128, dtype="float32")
    # W3
    W3 = init_weight(shape=(10, 128), fan_in=10, fan_out=10).astype("float32")
    # B3
    B3 = np.zeros(10, dtype="float32")

    return W1, B1, W2, B2, W3, B3
