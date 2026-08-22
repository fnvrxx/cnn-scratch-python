def predict_cnn(X, params, batch_size=512):
    preds = []

    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        logits, _ = forward_cnn(xb, params)
        preds.append(np.argmax(logits, axis=1))

    return np.concatenate(preds)


def evaluate_accuracy(X, y, params, batch_size=512):
    preds = predict_cnn(X, params, batch_size)
    return np.mean(preds == y)


def evaluate_cnn(X, y, params, batch_size=512):
    """
    Hitung loss dan accuracy untuk dataset X, y.
    """
    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]

        logits, _ = forward_cnn(xb, params)
        loss, _ = softmax_cross_entropy(logits, yb)

        bs = yb.shape[0]
        total_loss += loss * bs
        correct += np.sum(np.argmax(logits, axis=1) == yb)
        total += bs

    return total_loss / total, correct / total


def train_cnn(X, y, X_val, y_val, epochs=5, batch_size=128, lr=0.01):
    params = init_parameters_cnn()
    n = X.shape[0]

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)

        X_shuffled = X[idx]
        y_shuffled = y[idx]

        losses = []

        for start in range(0, n, batch_size):
            xb = X_shuffled[start : start + batch_size]
            yb = y_shuffled[start : start + batch_size]

            # Forward
            logits, layer_caches = forward_cnn(xb, params)

            # Loss dan gradient output
            loss, dz3 = softmax_cross_entropy(logits, yb)
            losses.append(loss)

            # Backward
            dW1, db1, dW2, db2, dW3, db3 = backward_cnn(dz3, caches, params)

            # Update parameter
            W1, b1, W2, b2, W3, b3 = params

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3

            params = (W1, b1, W2, b2, W3, b3)

        avg_loss = np.mean(losses)
        val_acc = evaluate_accuracy(X_val, y_val, params, batch_size=512)

        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")

    return params
