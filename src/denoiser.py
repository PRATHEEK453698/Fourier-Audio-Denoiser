import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import stft, istft


def _validate_audio(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("Audio must be mono. Convert stereo to mono first.")
    if audio.size < 32:
        raise ValueError("Audio is too short to process.")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite values.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    return audio


def denoise_audio(
    audio,
    sample_rate,
    noise_start=0.0,
    noise_end=0.35,
    strength=1.5,
    smoothing=1.0,
    floor=0.08,
    nperseg=1024,
    overlap_ratio=0.75,
):
    """
    STFT magnitude-gating denoiser.

    noise_start/noise_end identify a segment that should contain noise only.
    strength controls the noise threshold; higher values suppress more.
    smoothing controls mask smoothing over frequency/time bins.
    floor is the minimum retained amplitude gain.
    """
    audio = _validate_audio(audio, sample_rate)
    duration = len(audio) / sample_rate

    if not (0 <= noise_start < noise_end <= duration):
        raise ValueError(f"Noise interval must be inside the audio (0 to {duration:.2f} s).")
    if strength < 0.1 or strength > 5:
        raise ValueError("Strength must be between 0.1 and 5.")
    if smoothing < 0 or smoothing > 5:
        raise ValueError("Smoothing must be between 0 and 5.")
    if not 0 <= floor <= 1:
        raise ValueError("Floor must be between 0 and 1.")

    nperseg = min(int(nperseg), len(audio))
    nperseg = max(nperseg, 32)
    noverlap = min(int(nperseg * overlap_ratio), nperseg - 1)

    freqs, times, Z = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )

    magnitude = np.abs(Z)
    noise_frame_mask = (times >= noise_start) & (times <= noise_end)
    if not np.any(noise_frame_mask):
        raise ValueError("No STFT frames fall inside the chosen noise interval.")

    # Robust per-frequency noise estimate from the selected noise-only frames.
    noise_profile = np.median(magnitude[:, noise_frame_mask], axis=1)

    # A small frequency-dependent floor prevents near-zero bins dominating ratios.
    reference = max(float(np.max(noise_profile)), 1e-10)
    noise_profile = np.maximum(noise_profile, reference * 1e-4)

    # Soft spectral gate: preserve bins above the noise threshold, attenuate below.
    threshold = strength * noise_profile[:, None]
    raw_mask = np.clip(
        (magnitude - threshold) / np.maximum(magnitude, 1e-10),
        0.0,
        1.0,
    )

    if smoothing > 0:
        mask = gaussian_filter(
            raw_mask,
            sigma=(smoothing, smoothing),
            mode="nearest",
        )
    else:
        mask = raw_mask

    mask = np.clip(mask, floor, 1.0)

    # Apply real-valued gain to complex STFT coefficients; this retains their phase.
    cleaned_Z = Z * mask

    _, cleaned = istft(
        cleaned_Z,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        input_onesided=True,
        boundary=True,
        scaling="spectrum",
    )

    cleaned = cleaned[:len(audio)]
    if len(cleaned) < len(audio):
        cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))

    details = {
        "frequencies": freqs,
        "times": times,
        "original_stft": Z,
        "cleaned_stft": cleaned_Z,
        "noise_profile": noise_profile,
        "mask": mask,
        "noise_frames": int(np.sum(noise_frame_mask)),
        "mean_gain": float(np.mean(mask)),
        "min_gain": float(np.min(mask)),
        "max_gain": float(np.max(mask)),
        "original_rms": float(np.sqrt(np.mean(audio ** 2))),
        "cleaned_rms": float(np.sqrt(np.mean(cleaned ** 2))),
    }
    return cleaned, details


def stft_roundtrip(audio, sample_rate, nperseg=1024, overlap_ratio=0.75):
    """Diagnostic: STFT followed by inverse STFT without modifying coefficients."""
    audio = _validate_audio(audio, sample_rate)
    nperseg = min(max(32, int(nperseg)), len(audio))
    noverlap = min(int(nperseg * overlap_ratio), nperseg - 1)
    _, _, Z = stft(
        audio, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=noverlap, nfft=nperseg, detrend=False,
        return_onesided=True, boundary="zeros", padded=True,
        scaling="spectrum"
    )
    _, restored = istft(
        Z, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=noverlap, nfft=nperseg, input_onesided=True,
        boundary=True, scaling="spectrum"
    )
    restored = restored[:len(audio)]
    return restored
