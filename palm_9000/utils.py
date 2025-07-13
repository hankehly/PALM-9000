import io
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
from scipy.signal import resample_poly


def resample(
    audio: np.ndarray, original_sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    """
    This is the equivalent of calling:
    resample_poly(audio, target_sample_rate, original_sample_rate)

    But the program will use less compute resources if we reduce the
    ratio 44100:16000 to 441:160 with np.gcd (Greatest Common Divisor)
    which finds the largest integer that evenly divides two numbers.
    """
    gcd = np.gcd(original_sample_rate, target_sample_rate)
    return resample_poly(audio, target_sample_rate // gcd, original_sample_rate // gcd)


def play_audio(audio: bytes, sample_rate=16000, volume=1.0):
    """
    volume is a multiplier for the audio volume, so 1.0 is normal volume, 2.0 is double the volume, etc.
    Don't set it too high (>=3) or it will clip and distort the audio.
    """
    # Convert raw bytes to NumPy array of int16 samples
    pcm = np.frombuffer(audio, dtype=np.int16)

    # Apply volume gain (with clipping to int16 range)
    amplified = np.clip(pcm * volume, -32768, 32767).astype(np.int16)

    # Convert back to bytes
    amplified_bytes = amplified.tobytes()

    # Wrap PCM data in WAV headers in-memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(amplified_bytes)

    buffer.seek(0)

    # Play audio with PyAudio
    wf = wave.open(buffer, "rb")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
    )
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    pa.terminate()


def wait_until_device_available(device_index, timeout=2.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            sd.check_input_settings(device=device_index)
            return True
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(f"Mic still unavailable after {timeout} seconds.")


def remove_whitespace(text: str) -> str:
    """
    Removes all whitespace from the text.
    """
    return "".join(text.split())
