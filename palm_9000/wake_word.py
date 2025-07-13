import pvporcupine
import sounddevice as sd
from palm_9000 import settings


def wait_for_wake_word():
    """
    Waits for the wake word "computer" using Porcupine and sounddevice.
    This function blocks until the wake word is detected.
    """
    porcupine = pvporcupine.create(
        access_key=settings.picovoice_access_key.get_secret_value(),
        keywords=settings.porcupine_keywords,
    )

    try:
        with sd.InputStream(
            samplerate=porcupine.sample_rate,
            blocksize=porcupine.frame_length,
            dtype="int16",
            channels=1,
            device=settings.sounddevice_input_device,
        ) as stream:
            print("Listening for wake word...")

            while True:
                # Read one frame of audio
                audio_frame, _ = stream.read(porcupine.frame_length)
                # Flatten to 1D int16 numpy array
                pcm = audio_frame.flatten()
                # Process the audio frame with Porcupine
                result = porcupine.process(pcm)
                if result >= 0:
                    print(f"Detected {settings.porcupine_keywords[result]}")
                    break

        porcupine.delete()
        return True

    except KeyboardInterrupt:
        print("Stopped by user")
        porcupine.delete()
        return False
