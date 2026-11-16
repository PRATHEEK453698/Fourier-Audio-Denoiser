from pathlib import Path
import io

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from src.denoiser import denoise_audio, stft_roundtrip

st.set_page_config(page_title="Fourier Audio Lab", page_icon="🎧", layout="wide")
st.title("🎧 Fourier Audio Lab")
st.caption("Audio denoising using the Short-Time Fourier Transform and matrix-based spectral gating.")

with st.sidebar:
    st.header("Denoising controls")
    strength = st.slider("Noise suppression strength", 0.3, 4.0, 1.5, 0.1,
                         help="Increase to suppress more noise; excessive strength may remove desired audio.")
    smoothing = st.slider("Mask smoothing", 0.0, 3.0, 1.0, 0.25)
    floor = st.slider("Minimum retained gain", 0.0, 0.35, 0.08, 0.01,
                      help="A small floor can reduce musical artifacts but leaves some noise.")
    window_size = st.select_slider("STFT window size (samples)", options=[512, 1024, 2048, 4096], value=1024)
    passes = st.slider(
        "Repeated denoising passes",
        min_value=1,
        max_value=5,
        value=1,
        help="Each pass processes the previous pass's output. More passes may cause distortion."
    )

uploaded = st.file_uploader("Upload a WAV audio recording", type=["wav"])
if uploaded is None:
    st.info("Upload a WAV file to begin. For best results, choose a short interval containing background noise but no desired sound.")
    st.markdown("""
    **Project pipeline**
    1. STFT maps the waveform to a complex time–frequency matrix.
    2. The selected noise-only interval provides a robust median noise spectrum.
    3. A frequency/time suppression mask is calculated and smoothed.
    4. The mask is multiplied element-wise with the complex STFT.
    5. Inverse STFT reconstructs the cleaned waveform.
    """)
    st.stop()

try:
    data, sr = sf.read(uploaded, dtype="float64", always_2d=True)
except Exception as exc:
    st.error(f"Could not read this WAV file: {exc}")
    st.stop()

# Average channels to mono for this prototype.
audio = np.mean(data, axis=1)
duration = len(audio) / sr
st.write(f"**Sample rate:** {sr:,} Hz · **Duration:** {duration:.2f} s · **Channels:** {data.shape[1]}")

with st.expander("Choose the noise-only interval", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        noise_start = st.number_input("Noise starts (seconds)", min_value=0.0,
                                      max_value=max(0.0, duration - 0.02),
                                      value=0.05 if duration > 0.4 else 0.0, step=0.05)
    with c2:
        default_end = min(0.35, duration)
        noise_end = st.number_input("Noise ends (seconds)", min_value=0.01,
                                    max_value=float(duration),
                                    value=float(max(noise_start + 0.01, default_end)), step=0.05)
    st.caption("The selected segment should contain background noise without speech/music/desired signal. If the entire recording contains desired sound, choose the quietest available segment.")

if noise_end <= noise_start:
    st.error("Noise end must be after noise start.")
    st.stop()

if st.button("✨ Denoise audio", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing the noise spectrum and reconstructing audio..."):
            current_audio = audio
            info = None

            with st.spinner(f"Denoising audio ({passes} pass(es))..."):
                for pass_number in range(passes):
                    current_audio, info = denoise_audio(
                        current_audio,
                        sr,
                        noise_start=float(noise_start),
                        noise_end=float(noise_end),
                        strength=float(strength),
                        smoothing=float(smoothing),
                        floor=float(floor),
                        nperseg=int(window_size),
                    )

                clean = current_audio
            roundtrip = stft_roundtrip(audio, sr, nperseg=int(window_size))
            roundtrip_error = float(np.sqrt(np.mean((audio - roundtrip) ** 2)))
            buffer = io.BytesIO()
            sf.write(buffer, clean, sr, format="WAV", subtype="PCM_16")
            wav_bytes = buffer.getvalue()

        st.session_state["denoiser_result"] = {
            "clean": clean,
            "info": info,
            "wav_bytes": wav_bytes,
            "roundtrip_error": roundtrip_error,
            "sr": sr,
            "original": audio,
        }
    except Exception as exc:
        st.error(f"Denoising failed: {exc}")

result = st.session_state.get("denoiser_result")
if result is not None:
    original = result["original"]
    clean = result["clean"]
    info = result["info"]
    st.subheader("Results")
    a, b, c, d = st.columns(4)
    a.metric("Original RMS", f"{info['original_rms']:.5f}")
    b.metric("Cleaned RMS", f"{info['cleaned_rms']:.5f}")
    c.metric("Mean mask gain", f"{info['mean_gain']:.3f}")
    d.metric("STFT round-trip RMS error", f"{result['roundtrip_error']:.2e}")

    st.markdown("**Listen and compare**")
    left, right = st.columns(2)
    with left:
        st.write("Original")
        st.audio(uploaded.getvalue(), format="audio/wav")
    with right:
        st.write("Denoised")
        st.audio(result["wav_bytes"], format="audio/wav")

    st.download_button(
        "⬇️ Download denoised WAV",
        data=result["wav_bytes"],
        file_name="denoised_audio.wav",
        mime="audio/wav",
        use_container_width=True,
    )

    st.subheader("Time–frequency comparison")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    extent = [info["times"][0], info["times"][-1], info["frequencies"][0], info["frequencies"][-1]]
    eps = 1e-10
    axes[0].imshow(20 * np.log10(np.abs(info["original_stft"]) + eps),
                   origin="lower", aspect="auto", extent=extent, cmap="magma",
                   vmin=-100, vmax=-20)
    axes[0].set_title("Original STFT magnitude (dB)")
    axes[1].imshow(20 * np.log10(np.abs(info["cleaned_stft"]) + eps),
                   origin="lower", aspect="auto", extent=extent, cmap="magma",
                   vmin=-100, vmax=-20)
    axes[1].set_title("Denoised STFT magnitude (dB)")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, min(10000, result["sr"] / 2))
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("What the algorithm did")
    st.write(
        f"Used {info['noise_frames']} STFT frames in the selected noise interval. "
        f"The average mask gain was {info['mean_gain']:.3f}; 1.0 means no attenuation "
        "and values near 0 mean stronger attenuation."
    )
    st.caption("RMS reduction alone is not proof of noise removal. Listen to both versions and inspect the spectrogram; an unsuitable noise-only interval can make the result worse.")
