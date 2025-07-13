from pvrecorder import PvRecorder
import numpy as np
import pvporcupine
import sounddevice as sd

from palm_9000.vad import resample
from palm_9000.settings import settings


def wait_for_wake_word_sounddevice(device: int, input_rate: int):
    """
    Waits for the wake word using Porcupine and sounddevice.
    This function blocks until the wake word is detected.

    Default wake words include:
    'Alexa',
    'Americano',
    'Blueberry',
    'Bumblebee',
    'Computer',
    'Grapefruit',
    'Grasshopper',
    'Hey Google',
    'Hey Siri',
    'Jarvis',
    'Okay Google',
    'Picovoice',
    'Porcupine',
    'Terminator',
    """
    porcupine = pvporcupine.create(
        access_key=settings.picovoice_access_key.get_secret_value(),
        keyword_paths=[settings.porcupine_keyword_path],
        model_path=settings.porcupine_model_path,
    )

    frame_length = porcupine.frame_length  # usually 512
    target_rate = porcupine.sample_rate  # 16000 Hz
    resample_ratio = target_rate / input_rate

    # Number of input samples to resample into one Porcupine frame
    # ceil(512 / (16000/44100)) = 1412
    input_samples_per_frame = int(np.ceil(frame_length / resample_ratio))
    buffer = np.zeros(0, dtype=np.int16)

    print("Listening for wake word... (Press Ctrl+C to exit)")

    try:
        with sd.InputStream(
            samplerate=input_rate,
            blocksize=0,
            dtype="int16",
            channels=1,
            device=device,
        ) as stream:
            while True:
                audio_block, _ = stream.read(1024)
                buffer = np.concatenate([buffer, audio_block.flatten()])

                # Process as many full frames as we can
                while len(buffer) >= input_samples_per_frame:
                    chunk = buffer[:input_samples_per_frame]
                    buffer = buffer[input_samples_per_frame:]

                    resampled = resample(chunk, input_rate, target_rate)
                    if len(resampled) < frame_length:
                        continue

                    pcm = np.clip(resampled[:frame_length], -32768, 32767).astype(
                        np.int16
                    )

                    result = porcupine.process(pcm)
                    if result >= 0:
                        print("Wake word detected!")
                        return True

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
        return False

    finally:
        porcupine.delete()


def wait_for_wake_word_pvrecorder():
    porcupine = pvporcupine.create(
        access_key=settings.picovoice_access_key.get_secret_value(),
        keyword_paths=[settings.porcupine_keyword_path],
        model_path=settings.porcupine_model_path,
    )

    recorder = PvRecorder(
        device_index=-1,  # Use default input device
        frame_length=porcupine.frame_length,
    )

    print("Listening for wake word... (Press Ctrl+C to exit)")

    try:
        recorder.start()

        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)
            if result >= 0:
                print("Wake word detected!")
                return True

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
        return False

    finally:
        recorder.stop()
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    device = sd.query_devices(kind="input")
    print(
        f"Using input device: {device['name']} at sample rate: {device['default_samplerate']}"
    )
    wait_for_wake_word(device["index"], int(device["default_samplerate"]))
