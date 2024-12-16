import numpy as np
from cost_function import compute_cost
from gradient_function import compute_gradient
from gradient_descent_function import compute_gradient_descent


def univariate():
    x = np.array([1.0, 2.0])
    y = np.array([300.0, 500.0])
    w_in = 0
    b_in = 0
    alpha = 1.0e-1
    num_iters = 10000

    return compute_gradient_descent(
        x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient
    )


w_final, b_final, J_hist, p_hist = univariate()
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
