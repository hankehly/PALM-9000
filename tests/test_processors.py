from unittest.mock import AsyncMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from palm_9000.processors import AudioRecordingControlProcessor


@pytest.fixture
def audio_buffer():
    buffer = AsyncMock()
    buffer.start_recording = AsyncMock()
    buffer.stop_recording = AsyncMock()
    return buffer


@pytest.fixture
def processor(audio_buffer, monkeypatch):
    proc = AudioRecordingControlProcessor(audio_buffer)
    # Detach from a real pipeline: capture pushes instead of forwarding.
    pushed = []
    monkeypatch.setattr(
        proc, "push_frame", AsyncMock(side_effect=lambda f, d: pushed.append((f, d)))
    )
    proc.pushed = pushed
    return proc


async def test_bot_started_speaking_starts_recording(processor, audio_buffer):
    await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    audio_buffer.start_recording.assert_awaited_once()
    audio_buffer.stop_recording.assert_not_awaited()


@pytest.mark.parametrize(
    "frame",
    [
        BotStoppedSpeakingFrame(),
        CancelFrame(),
        EndFrame(),
        ErrorFrame(error="boom"),
    ],
    ids=["bot_stopped", "cancel", "end", "error"],
)
async def test_terminal_frames_stop_recording(processor, audio_buffer, frame):
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    audio_buffer.stop_recording.assert_awaited_once()
    audio_buffer.start_recording.assert_not_awaited()


async def test_unrelated_frames_touch_neither(processor, audio_buffer):
    await processor.process_frame(TextFrame(text="hi"), FrameDirection.DOWNSTREAM)
    audio_buffer.start_recording.assert_not_awaited()
    audio_buffer.stop_recording.assert_not_awaited()


@pytest.mark.parametrize(
    "frame",
    [
        BotStartedSpeakingFrame(),
        BotStoppedSpeakingFrame(),
        TextFrame(text="passthrough"),
    ],
    ids=["started", "stopped", "text"],
)
async def test_every_frame_is_forwarded(processor, frame):
    """The processor is a tap, not a filter - nothing may be swallowed."""
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    assert [f for f, _ in processor.pushed] == [frame]


async def test_direction_is_preserved(processor):
    await processor.process_frame(TextFrame(text="up"), FrameDirection.UPSTREAM)
    assert processor.pushed[0][1] is FrameDirection.UPSTREAM


async def test_full_speaking_cycle(processor, audio_buffer):
    await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    audio_buffer.start_recording.assert_awaited_once()
    audio_buffer.stop_recording.assert_awaited_once()
    assert len(processor.pushed) == 2


def test_holds_the_audio_buffer_it_controls(audio_buffer):
    proc = AudioRecordingControlProcessor(audio_buffer)
    assert proc._audio_buffer is audio_buffer
