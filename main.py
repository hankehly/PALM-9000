import datetime

import numpy as np
import scipy.io.wavfile
import webrtcvad

from palm_9000.llm import run_llm
from palm_9000.settings import settings
from palm_9000.speech_to_text import speech_to_text, STT_SAMPLE_RATE
from palm_9000.text_to_speech import text_to_speech
from palm_9000.utils import play_audio, wait_until_device_available
from palm_9000.wake_word import wait_for_wake_word
from palm_9000.vad import (
    microphone_audio_frame_generator,
    resample_frames,
    vad_collector,
    Frame,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, StateGraph

VAD_FRAME_DURATION_MS = 30
VAD_SAMPLE_RATE = 32000
THREAD_ID = "1"


def main():
    graph = StateGraph(state_schema=MessagesState)
    graph.add_node("run_llm", run_llm)
    graph.set_entry_point("run_llm")

    checkpointer = InMemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)

    vad = webrtcvad.Vad(settings.vad_mode)

    while True:
        print("🌴 Waiting up to 5 seconds for microphone...")
        wait_until_device_available(settings.input_device, timeout=5.0)

        print("🌴 Waiting for wake word...")
        if not wait_for_wake_word():
            break

        print("🌴 Waiting up to 5 seconds for microphone...")
        wait_until_device_available(settings.input_device, timeout=5.0)

        # Generate audio frames from the microphone input
        raw_frame_generator = microphone_audio_frame_generator(
            frame_duration_ms=VAD_FRAME_DURATION_MS,
            sample_rate=settings.sample_rate,
            device=settings.input_device,
        )

        # Resample the audio frames to the VAD sample rate
        resampled_frames = resample_frames(
            raw_frame_generator,
            original_sample_rate=settings.sample_rate,
            target_sample_rate=VAD_SAMPLE_RATE,
        )

        # Collect voiced audio chunks using VAD
        # The sample rate of each chunk will be VAD_SAMPLE_RATE
        voiced_audio_generator = vad_collector(
            sample_rate=VAD_SAMPLE_RATE,
            frame_duration_ms=VAD_FRAME_DURATION_MS,
            padding_duration_ms=500,
            vad=vad,
            frames=resampled_frames,
            silence_timeout=settings.silence_timeout,
        )

        print("🌴 Collecting voiced audio chunks... (Start speaking)")
        voiced_frames_vad_sr = []
        for chunk in voiced_audio_generator:
            print(f"  Voiced audio chunk of length {len(chunk)} bytes detected.")
            voiced_frames_vad_sr.append(
                Frame(bytes=chunk, timestamp=None, duration=None)
            )

        # Resample back down to STT_SAMPLE_RATE
        print("🌴 Resampling audio chunks to speech-to-text sample rate...")
        voiced_frames_stt_sr = resample_frames(
            voiced_frames_vad_sr,
            original_sample_rate=VAD_SAMPLE_RATE,
            target_sample_rate=STT_SAMPLE_RATE,
        )

        # Combine all voiced frames into a single byte string for STT processing
        voiced_frame_bytes_stt_sr = b"".join(
            frame.bytes for frame in voiced_frames_stt_sr
        )

        # Export to wav file for debugging
        scipy.io.wavfile.write(
            f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
            # settings.sample_rate,
            STT_SAMPLE_RATE,
            np.frombuffer(voiced_frame_bytes_stt_sr, dtype=np.int16),
        )

        print("🌴 Processing audio chunks for speech-to-text...")
        speech_to_text_result = speech_to_text(voiced_frame_bytes_stt_sr)

        print(f"🌴 Speech to text result: {speech_to_text_result}")
        if not speech_to_text_result:
            print("🌴 No speech detected, waiting for wake word again...")
            continue

        state = compiled_graph.invoke(
            input={"messages": [speech_to_text_result]},
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        llm_response = state["messages"][-1].content
        print(f"🌴 LLM response: {llm_response}")

        if not llm_response:
            print("🌴 No response from LLM, waiting for wake word again...")
            continue

        print("🌴 Converting text to speech...")
        text_to_speech_result = text_to_speech(llm_response)

        print("🌴 Playing audio response...")
        play_audio(text_to_speech_result)


if __name__ == "__main__":
    main()
