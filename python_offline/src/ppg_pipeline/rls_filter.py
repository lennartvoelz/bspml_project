import numpy as np

def create_tap_delay_input(acc_input, order):
    """
    Builds a matrix of delayed inputs for each timestep.
    acc_input: (N,) or (N, D)
    Returns: (N, order * D)
    """
    N = len(acc_input)
    if acc_input.ndim == 1:
        acc_input = acc_input.reshape(-1, 1)  # Make it (N, 1)
    D = acc_input.shape[1]
    delayed = np.zeros((N, order * D))
    for i in range(order, N):
        delayed[i] = acc_input[i - order:i].reshape(-1)
    return delayed

class RLSFilter:
    def __init__(self, num_features, lambda_=0.99, delta=1.0):
        self.n = num_features
        self.lambda_ = lambda_
        self.w = np.zeros((self.n, 1))  # Weight vector
        self.P = (1.0 / delta) * np.eye(self.n)  # Inverse correlation matrix

    def update(self, x, d):
        x = x.reshape(-1, 1)
        x_norm = np.linalg.norm(x) + 1e-8  # avoid division by zero
        x /= x_norm

        d = np.array([[d]])
        pi = self.P @ x
        denom = self.lambda_ + x.T @ pi
        if denom == 0 or np.isnan(denom):
            return float(d)  # skip update

        k = pi / denom
        e = d - self.w.T @ x
        self.w += k @ e
        self.P = (self.P - k @ x.T @ self.P) / self.lambda_
        # self.P = np.clip(self.P, -1e6, 1e6) # clip

        y = float(self.w.T @ x)
        return y

    
def rls_filter(ppg, acc_input, forgetting_factor=0.99, order=4):
    """
    RLS filter using a tap-delay line of ACC input
    """
    # Create delayed input matrix
    delayed_input = create_tap_delay_input(acc_input, order)
    num_features = delayed_input.shape[1]

    filtered_ppg = np.zeros_like(ppg)
    rls = RLSFilter(num_features=num_features, lambda_=forgetting_factor, delta=100.0)

    for i in range(order, len(ppg)):
        x = delayed_input[i]
        artifact = rls.update(x, ppg[i])
        filtered_ppg[i] = ppg[i] - artifact

    # Optionally fill first 'order' entries with original signal
    filtered_ppg[:order] = ppg[:order]

    return filtered_ppg
