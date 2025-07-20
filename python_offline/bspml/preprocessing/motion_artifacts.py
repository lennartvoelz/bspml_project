import numpy as np
from typing import Optional


def combine_accelerometer_channels(acc_signals: np.ndarray) -> np.ndarray:
    """
    Combine accelerometer channels using Euclidean norm.

    Args:
        acc_signals: Accelerometer signals [n_samples, n_channels]
                    Can be 1D (single channel) or 2D (multiple channels)

    Returns:
        Combined accelerometer signal [n_samples, 1]
    """
    if acc_signals.ndim == 1:
        # Single channel - reshape to column vector
        return acc_signals.reshape(-1, 1)
    elif acc_signals.ndim == 2:
        if acc_signals.shape[1] == 1:
            # Already single channel
            return acc_signals
        else:
            # Multiple channels - compute Euclidean norm
            magnitude = np.sqrt(np.sum(acc_signals**2, axis=1))
            return magnitude.reshape(-1, 1)
    else:
        raise ValueError(
            f"Accelerometer signals must be 1D or 2D, got {acc_signals.ndim}D"
        )


class RLSFilter:
    """RLS adaptive filter for motion artifact removal using accelerometer reference."""

    def __init__(
        self,
        filter_order: int = 8,
        forgetting_factor: float = 0.99,
        regularization: float = 1e-4,
    ):
        """
        Initialize RLS filter.

        Args:
            filter_order: Order of the adaptive filter
            forgetting_factor: Forgetting factor (0 < lambda <= 1)
            regularization: Regularization parameter for numerical stability
        """
        self.filter_order = filter_order
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization

        # Initialize filter coefficients and covariance matrix
        self.weights = np.zeros(filter_order)
        self.P = np.eye(filter_order) / regularization

    def reset(self):
        """Reset filter state."""
        self.weights = np.zeros(self.filter_order)
        self.P = np.eye(self.filter_order) / self.regularization

    def update(self, reference_vector: np.ndarray, desired_signal: float) -> float:
        """
        Update filter with one sample.

        Args:
            reference_vector: Reference signal vector (accelerometer data)
            desired_signal: Desired signal sample (PPG)

        Returns:
            Filtered output sample
        """
        if len(reference_vector) != self.filter_order:
            raise ValueError(f"Reference vector must have length {self.filter_order}")

        # Calculate filter output
        output = np.dot(self.weights, reference_vector)

        # Calculate error
        error = desired_signal - output

        # Update covariance matrix
        k = self.P @ reference_vector
        alpha = 1 / (self.forgetting_factor + reference_vector.T @ k)
        self.P = (self.P - alpha * np.outer(k, k)) / self.forgetting_factor

        # Update weights
        self.weights += alpha * error * k

        return error  # Return cleaned signal (error signal)


def rls_filter(
    ppg_signal: np.ndarray,
    acc_signals: np.ndarray,
    filter_order: int = 15,
    forgetting_factor: float = 0.97,
    delay_compensation: Optional[int] = None,
    auto_delay_detection: bool = True,
) -> np.ndarray:
    """
    Apply RLS adaptive filtering for motion artifact removal using accelerometer reference.

    Args:
        ppg_signal: Input PPG signal
        acc_signals: Accelerometer signals (N x 1 or N x 3 for x, y, z axes)
        filter_order: Adaptive filter order
        forgetting_factor: RLS forgetting factor
        delay_compensation: Manual delay in samples (positive = ACC leads PPG)
        auto_delay_detection: Whether to automatically detect optimal delay

    Returns:
        Motion artifact corrected PPG signal
    """
    if len(ppg_signal) == 0:
        raise ValueError("PPG signal cannot be empty")

    if acc_signals.ndim not in [1, 2]:
        raise ValueError("Accelerometer signals must be 1D or 2D array")

    # Combine accelerometer channels using Euclidean norm
    acc_combined = combine_accelerometer_channels(acc_signals)

    if len(ppg_signal) != acc_combined.shape[0]:
        raise ValueError("PPG and accelerometer signals must have same length")

    n_samples, n_axes = acc_combined.shape

    # Apply delay compensation if requested
    if auto_delay_detection:
        delay_compensation = detect_optimal_delay(ppg_signal, acc_combined.flatten())
        print(f"Auto-detected delay: {delay_compensation} samples")

    if delay_compensation is not None and delay_compensation != 0:
        acc_combined = apply_delay_compensation(acc_combined, delay_compensation)

    # Create reference signal matrix by combining accelerometer signal
    # and its delayed versions to capture motion dynamics
    reference_matrix = create_reference_matrix(acc_combined, filter_order)

    # Initialize RLS filter
    rls = RLSFilter(filter_order, forgetting_factor)

    # Process signal sample by sample
    cleaned_signal = np.zeros_like(ppg_signal)

    for i in range(filter_order, n_samples):
        ref_vector = reference_matrix[i, :]
        cleaned_signal[i] = rls.update(ref_vector, ppg_signal[i])

    # Copy initial samples (before filter has enough history)
    cleaned_signal[:filter_order] = ppg_signal[:filter_order]

    return cleaned_signal


def create_reference_matrix(acc_signals: np.ndarray, filter_order: int) -> np.ndarray:
    """
    Create reference signal matrix from accelerometer data.

    This function creates a matrix of reference signals using time-delayed
    versions of the accelerometer signal.

    Args:
        acc_signals: Accelerometer signals [n_samples, 1] (single combined channel)
        filter_order: Filter order (number of taps/delays)

    Returns:
        Reference matrix [n_samples, filter_order]
    """
    n_samples, n_channels = acc_signals.shape

    if n_channels != 1:
        raise ValueError(
            f"Expected single channel accelerometer data, got {n_channels} channels"
        )

    # Create time-delayed versions of the accelerometer signal
    reference_matrix = np.zeros((n_samples, filter_order))

    # Extract the single channel
    acc_signal = acc_signals[:, 0]

    # Fill reference matrix with time-delayed versions
    for delay in range(filter_order):
        if delay == 0:
            # No delay - current sample
            reference_matrix[:, delay] = acc_signal
        else:
            # Delayed version - shift signal by 'delay' samples
            reference_matrix[delay:, delay] = acc_signal[:-delay]
            # Fill initial samples with first value (zero-padding alternative)
            reference_matrix[:delay, delay] = acc_signal[0]

    return reference_matrix


def detect_optimal_delay(
    ppg_signal: np.ndarray, acc_signal: np.ndarray, max_delay: int = 20
) -> int:
    """
    Detect optimal delay between PPG and accelerometer signals.

    Args:
        ppg_signal: PPG signal
        acc_signal: Accelerometer signal (1D)
        max_delay: Maximum delay to test (in samples)

    Returns:
        Optimal delay in samples (positive = ACC leads PPG)
    """
    from scipy import signal as scipy_signal

    # Extract high-frequency components for better correlation
    fs = 50.0  # Assume 64 Hz sampling rate
    nyquist = fs / 2
    high_cutoff = 1.0  # Hz

    if high_cutoff < nyquist:
        b, a = scipy_signal.butter(4, high_cutoff / nyquist, btype="high")
        ppg_high = scipy_signal.filtfilt(b, a, ppg_signal)
        acc_high = scipy_signal.filtfilt(b, a, acc_signal)
    else:
        ppg_high = ppg_signal.copy()
        acc_high = acc_signal.copy()

    # Test different delays
    delay_range = np.arange(-max_delay, max_delay + 1)
    correlations = []

    for delay in delay_range:
        if delay == 0:
            corr = np.corrcoef(ppg_high, acc_high)[0, 1]
        elif delay > 0:
            # ACC leads PPG
            if delay < len(acc_high):
                corr = np.corrcoef(ppg_high[delay:], acc_high[:-delay])[0, 1]
            else:
                corr = 0
        else:
            # PPG leads ACC
            delay_abs = abs(delay)
            if delay_abs < len(ppg_high):
                corr = np.corrcoef(ppg_high[:-delay_abs], acc_high[delay_abs:])[0, 1]
            else:
                corr = 0

        correlations.append(abs(corr))  # Use absolute correlation

    # Find delay with maximum absolute correlation
    best_idx = np.argmax(correlations)
    optimal_delay = delay_range[best_idx]

    return optimal_delay


def apply_delay_compensation(acc_signals: np.ndarray, delay: int) -> np.ndarray:
    """
    Apply delay compensation to accelerometer signals.

    Args:
        acc_signals: Accelerometer signals [n_samples, n_channels]
        delay: Delay in samples (positive = shift ACC forward, negative = shift backward)

    Returns:
        Delay-compensated accelerometer signals
    """
    if delay == 0:
        return acc_signals

    compensated = np.zeros_like(acc_signals)

    if delay > 0:
        # Shift ACC forward (ACC leads PPG)
        compensated[delay:] = acc_signals[:-delay]
        # Fill initial samples with first value
        compensated[:delay] = acc_signals[0]
    else:
        # Shift ACC backward (PPG leads ACC)
        delay_abs = abs(delay)
        compensated[:-delay_abs] = acc_signals[delay_abs:]
        # Fill final samples with last value
        compensated[-delay_abs:] = acc_signals[-1]

    return compensated
