import numpy as np
from typing import Optional
import pywt


def wavelet_detrend(
    signal: np.ndarray,
    wavelet: str = 'db4',
    levels: Optional[int] = None,
    mode: str = 'symmetric'
) -> np.ndarray:
    """
    Remove baseline drift from PPG signal using wavelet decomposition.

    Args:
        signal: Input PPG signal
        wavelet: Wavelet type (default: 'db4')
        levels: Decomposition levels (default: auto-calculated)
        mode: Signal extension mode

    Returns:
        Detrended PPG signal
    """
    if len(signal) == 0:
        raise ValueError("Input signal cannot be empty")

    if not np.isfinite(signal).all():
        raise ValueError("Input signal contains non-finite values")

    if levels is None:
        levels = min(6, int(np.log2(len(signal))))

    coeffs = pywt.wavedec(signal, wavelet, level=levels, mode=mode)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove low-frequency components
    detrended_signal = pywt.waverec(coeffs, wavelet, mode=mode)
    if len(detrended_signal) != len(signal):
        detrended_signal = detrended_signal[:len(signal)]

    return detrended_signal
