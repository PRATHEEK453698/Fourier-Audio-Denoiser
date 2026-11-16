# Fourier Audio Lab — STFT-Based Audio Denoiser

A Python-based audio denoising application that uses the **Short-Time Fourier Transform (STFT)** and spectral gating to reduce background noise from audio recordings.

The project converts an audio waveform into a time-frequency representation, estimates background noise from a user-selected noise-only segment, applies a soft spectral mask, and reconstructs the denoised audio.

## Features

- Upload WAV audio recordings through a Streamlit interface.
- Select a noise-only segment for noise profile estimation.
- Adjust denoising strength.
- Apply spectral gating with Gaussian smoothing.
- Preserve phase information while modifying spectral magnitudes.
- Reconstruct and listen to the processed audio.
- Save the denoised output separately from the original recording.
- Command-line execution for audio processing.

## Technology Stack

- **Python** — Core implementation
- **NumPy** — Numerical and matrix operations
- **SciPy** — Signal processing and STFT operations
- **Streamlit** — Interactive frontend
- **SoundFile** — Audio file handling

## Project Structure

```text
Fourier-Audio-Denoiser/
│
├── app.py
├── run_denoising.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   └── denoiser.py
│
└── data/
    └── test_samples/
        └── PLACE_CATS_CAN_WAV_HERE.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/PRATHEEK453698/Fourier-Audio-Denoiser.git
cd Fourier-Audio-Denoiser
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Application

### Streamlit Interface

Launch the interactive application:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Upload a WAV recording, select the noise-only interval, adjust the denoising strength, and process the audio.

### Command-Line Execution

Place the original WAV recording inside:

```text
data/test_samples/catscan.wav
```

Run:

```bash
python run_denoising.py
```

The denoised recording is saved separately as:

```text
data/test_samples/denoised_catscan.wav
```

## How It Works

The denoising pipeline consists of the following stages:

1. **STFT Transformation:** Converts the input waveform into a complex time-frequency matrix, representing the signal across time and frequency.

2. **Noise Profile Estimation:** Estimates the background noise spectrum using the median magnitude spectrum of a user-selected noise-only segment.

3. **Spectral Mask Generation:** Computes a soft mask based on the observed magnitude spectrum, estimated noise profile, and selected denoising strength.

4. **Mask Smoothing:** Applies Gaussian smoothing to reduce abrupt variations across time and frequency.

5. **Spectral Filtering:** Multiplies the complex STFT coefficients by the soft mask, modifying their magnitudes while retaining their phase wherever the mask is nonzero.

6. **Inverse STFT:** Reconstructs the time-domain audio waveform from the processed spectrogram.

## Mathematical Foundation

The input audio signal is modelled as:

\[
y(t)=x(t)+n(t)
\]

Where:

- \(y(t)\) is the observed noisy signal.
- \(x(t)\) is the desired clean signal.
- \(n(t)\) is the background noise.

The STFT transforms the signal into the time-frequency domain:

\[
X(m,k)=\sum_n x[n]w[n-mH]e^{-j2\pi kn/N}
\]

Where:

- \(w[n]\) is the analysis window.
- \(H\) is the hop length.
- \(N\) is the FFT size.
- \(m\) and \(k\) represent time frames and frequency bins.

A soft spectral mask is then applied to suppress frequency components identified as noise.

## Limitations

This project implements a classical spectral-gating approach. It works best with reasonably stationary background noise and a representative noise-only segment.

It cannot perfectly separate desired audio and noise when they overlap in time and frequency. Excessive denoising strength may also introduce audio artifacts or remove parts of the desired signal.

The denoising result therefore depends on the quality of the selected noise interval and the chosen processing parameters.

## Future Improvements

- Adaptive noise profile estimation.
- Improved handling of non-stationary noise.
- More advanced spectral filtering techniques.
- Better preservation of speech and other desired audio components.
- Performance and reconstruction quality improvements.

## Project Objective

The objective of this project is to explore the application of Fourier analysis, matrix-based signal representation, and digital signal processing techniques to practical audio denoising.

