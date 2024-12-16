def compute_cost_linear(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        # f_wb = np.dot(x, w) + b
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost
