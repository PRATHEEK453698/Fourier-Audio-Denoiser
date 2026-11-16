from pathlib import Path
import numpy as np
import soundfile as sf

from src.denoiser import denoise_audio

project_root = Path(__file__).parent
input_path = project_root / "data" / "test_samples" / "catscan.wav"
output_path = project_root / "data" / "test_samples" / "denoised_catscan.wav"

audio, sr = sf.read(input_path, dtype="float64", always_2d=True)
audio = np.mean(audio, axis=1)
duration = len(audio) / sr

noise_start = 0.05
noise_end = min(0.35, duration)
if noise_end <= noise_start:
    raise ValueError("Recording is too short for the configured noise interval.")

clean, details = denoise_audio(
    audio,
    sr,
    noise_start=noise_start,
    noise_end=noise_end,
    strength=1.5,
    smoothing=1.0,
    floor=0.08,
    nperseg=1024,
)

sf.write(output_path, clean, sr, subtype="PCM_16")

print(f"Input: {input_path.name}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Noise interval: {noise_start:.2f}–{noise_end:.2f} seconds")
print(f"Noise frames used: {details['noise_frames']}")
print(f"Original RMS: {details['original_rms']:.6f}")
print(f"Denoised RMS: {details['cleaned_rms']:.6f}")
print(f"Mean mask gain: {details['mean_gain']:.4f}")
print(f"Gain range: {details['min_gain']:.3f}–{details['max_gain']:.3f}")
print(f"Saved: {output_path}")
