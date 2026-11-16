# Fourier Audio Lab — STFT + Linear Algebra Denoiser

## Setup (Windows / PyCharm terminal)

From the project folder, activate your existing virtual environment if needed, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Launch the presentation interface

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Upload a WAV recording.

## Run the command-line version

Put the original WAV here:

`data/test_samples/catscan.wav`

Then run:

```powershell
python run_denoising.py
```

The result is saved separately as `data/test_samples/denoised_catscan.wav`.

## How it works

- STFT maps the waveform into a complex time-frequency matrix.
- A median magnitude spectrum is estimated from a user-selected noise-only segment.
- A soft mask is computed using the magnitude and noise threshold.
- Gaussian smoothing reduces abrupt changes across frequency and time.
- The mask multiplies the complex STFT coefficients, retaining phase.
- Inverse STFT reconstructs the output waveform.

## Important

This is a classical spectral-gating prototype, best suited to reasonably stationary background noise when a representative noise-only segment is available. It cannot perfectly separate desired sound and noise that overlap in time and frequency. Try several strengths and noise intervals; stronger is not always better.
