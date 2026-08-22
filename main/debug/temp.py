# x = np.array(
#     [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]], dtype="float32"
# )


# N = GAMBAR -> 1
# C = CHANNEL/WARNA -> 1
# H = HEIGHT -> 4
# W = WEIGHT -> 4
# ARRAY x yang diatas itu merupakan hasil visualnya

# x = x.reshape(1, 1, 4, 4)  # N=1, C=1, H=4, W=4
# print("x: ", x)
# print("shape x:", x.shape)

# # W -> weight untuk kernel
# W = np.array([[[1, 0], [0, 1]]], dtype="float32")

# W = W.reshape(1, 1, 2, 2)  # eksperimen pertama

# b = np.array([0], dtype="float32")  # bias

# cols = im2col(x, kernel_size=(2, 2), stride=1)

# Z = conv2d_forward_ujicoba(x, W, b, stride=1, pad=0)

# # print("cols:")
# # print(cols)
# # print("shape:", cols.shape)
# print("w:", W.shape)
# print(Z)
# print(
#     Z.shape
# )  # output shape ini seperti N,F,H,W (IMAGE : 1, 2 FEATURE MAP, 3 PIXEL UNTUK HEIGHT, 3 PIXEL UNTUK WEIGHT)
