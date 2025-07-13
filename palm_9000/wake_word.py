import numpy as np
import pvporcupine
import sounddevice as sd
from palm_9000.vad import resample
from palm_9000.settings import settings


def wait_for_wake_word(device: int, sample_rate: int):
    """
    Waits for the wake word using Porcupine and sounddevice.
    This function blocks until the wake word is detected.
    """
    porcupine = pvporcupine.create(
        access_key=settings.picovoice_access_key.get_secret_value(),
        keywords=[settings.porcupine_keyword],
        # keyword_paths=[settings.porcupine_keyword_path],
        # model_path=settings.porcupine_model_path,
    )

    try:
        # Determine how many input samples give us enough to downsample to one Porcupine frame
        input_samples_per_frame = int(
            porcupine.frame_length * sample_rate / porcupine.sample_rate
        )

        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=input_samples_per_frame,
            dtype="int16",
            channels=1,
            device=device,
        ) as stream:
            print("Listening for wake word...")

            while True:
                block, _ = stream.read(input_samples_per_frame)
                block = block.flatten()
                # Downsample from 44100 Hz to 16000 Hz
                # Sample rate values must be ints
                downsampled = resample(
                    block, sample_rate, porcupine.sample_rate
                )
                pcm = np.clip(downsampled, -32768, 32767).astype(np.int16)
                if len(pcm) != porcupine.frame_length:
                    print(f"Expected {porcupine.frame_length} samples, got {len(pcm)}")
                    continue  # Skip incomplete frames
                result = porcupine.process(pcm)
                if result >= 0:
                    print(f"Detected {settings.porcupine_keyword}")
                    break

        porcupine.delete()
        return True

    except KeyboardInterrupt:
        print("Stopped by user")
        porcupine.delete()
        return False


if __name__ == "__main__":
    if wait_for_wake_word():
        print("Wake word detected!")
    else:
        print("Wake word detection stopped.")
