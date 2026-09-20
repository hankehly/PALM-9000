"""Tests for the pipeline wiring in main.py.

These are regression tests for the two failures that took PALM-9000 down:

1. Without a context aggregator pair AND an LLMRunFrame kickoff,
   GeminiLiveLLMService never sets _ready_for_realtime_input and silently
   discards every frame of user audio.
2. WorkerRunner.add_workers is a coroutine; calling it without await
   registers no worker and the pipeline connects to nothing.

Both fail silently at runtime, so they are asserted explicitly here.
"""

import inspect
from unittest.mock import MagicMock

import pytest

import main as main_module


def _settings(**overrides):
    """A real Settings object with wake-word fields overridden."""
    from palm_9000.settings import Settings

    base = {
        "google_api_key": "k",
        "wake_word_enabled": False,
        "wake_word_model_path": "models/wakeword/hey_livekit.onnx",
        "wake_word_threshold": 0.5,
        "wake_silence_timeout_secs": 30.0,
        "wake_word_hop_samples": 1280,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakeEventEmitter:
    """Captures handlers registered via the @x.event_handler("name") decorator."""

    def __init__(self):
        self.handlers: dict[str, list] = {}

    def event_handler(self, name):
        def decorator(func):
            self.handlers.setdefault(name, []).append(func)
            return func

        return decorator


class FakeAudioBuffer(FakeEventEmitter):
    instances: list["FakeAudioBuffer"] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        FakeAudioBuffer.instances.append(self)


class FakeWorker(FakeEventEmitter):
    instances: list["FakeWorker"] = []

    def __init__(self, pipeline, **kwargs):
        super().__init__()
        self.pipeline = pipeline
        self.kwargs = kwargs
        self.queued: list = []
        FakeWorker.instances.append(self)

    async def queue_frames(self, frames):
        self.queued.extend(frames)


class FakeHeart:
    instances: list["FakeHeart"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = 0
        self.stopped = 0
        self.audio_chunks: list[bytes] = []
        FakeHeart.instances.append(self)

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop()

    def process_audio(self, audio):
        self.audio_chunks.append(audio)


class FakeRunner:
    instances: list["FakeRunner"] = []

    def __init__(self):
        self.added: list = []
        self.ran = 0
        self.run_error: Exception | None = None
        FakeRunner.instances.append(self)

    async def add_workers(self, *workers):
        self.added.extend(workers)

    async def run(self):
        self.ran += 1
        if self.run_error:
            raise self.run_error


@pytest.fixture
def wired(monkeypatch):
    """Replace every external collaborator of main() with a fake."""
    FakeHeart.instances.clear()
    FakeRunner.instances.clear()
    FakeAudioBuffer.instances.clear()
    FakeWorker.instances.clear()

    captured = {}

    def fake_pipeline(processors):
        captured["processors"] = processors
        return MagicMock(name="Pipeline")

    def fake_transport(params):
        captured["transport_params"] = params
        transport = MagicMock(name="LocalAudioTransport")
        transport.input.return_value = "TRANSPORT_IN"
        transport.output.return_value = "TRANSPORT_OUT"
        return transport

    def fake_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return "LLM"

    def fake_aggregator_pair(context, **kwargs):
        captured["context"] = context
        pair = MagicMock(name="LLMContextAggregatorPair")
        pair.user.return_value = "AGG_USER"
        pair.assistant.return_value = "AGG_ASSISTANT"
        return pair

    monkeypatch.setattr(main_module, "AudioBufferProcessor", FakeAudioBuffer)
    monkeypatch.setattr(main_module, "Max7219AmplitudeHeart", FakeHeart)
    monkeypatch.setattr(main_module, "PipelineWorker", FakeWorker)
    monkeypatch.setattr(main_module, "WorkerRunner", FakeRunner)
    monkeypatch.setattr(main_module, "Pipeline", fake_pipeline)
    monkeypatch.setattr(main_module, "LocalAudioTransport", fake_transport)
    monkeypatch.setattr(main_module, "GeminiLiveLLMService", fake_llm)
    monkeypatch.setattr(main_module, "LLMContextAggregatorPair", fake_aggregator_pair)
    monkeypatch.setattr(
        main_module,
        "AudioRecordingControlProcessor",
        lambda buffer: "RECORDING_CONTROL",
    )
    return captured


class TestRealtimeGateRegression:
    """The bug that made the bot silent: audio dropped before reaching Gemini."""

    async def test_queues_an_llm_run_frame(self, wired):
        await main_module.main()

        worker = _worker()
        assert len(worker.queued) == 1
        assert isinstance(worker.queued[0], main_module.LLMRunFrame)

    async def test_pipeline_contains_both_context_aggregators(self, wired):
        await main_module.main()

        processors = wired["processors"]
        assert "AGG_USER" in processors
        assert "AGG_ASSISTANT" in processors

    async def test_user_aggregator_precedes_the_llm(self, wired):
        """It must sit upstream so the context frame reaches the service."""
        await main_module.main()

        processors = wired["processors"]
        assert processors.index("AGG_USER") < processors.index("LLM")

    async def test_assistant_aggregator_is_last(self, wired):
        await main_module.main()
        assert wired["processors"][-1] == "AGG_ASSISTANT"

    async def test_context_is_an_llm_context(self, wired):
        await main_module.main()
        assert isinstance(wired["context"], main_module.LLMContext)


class TestWorkerRegistrationRegression:
    """The bug that stopped it connecting: add_workers called without await."""

    def test_add_workers_is_a_coroutine_function(self):
        from pipecat.workers.runner import WorkerRunner

        assert inspect.iscoroutinefunction(WorkerRunner.add_workers)

    async def test_the_worker_is_actually_registered(self, wired):
        await main_module.main()

        runner = FakeRunner.instances[0]
        assert len(runner.added) == 1
        assert isinstance(runner.added[0], FakeWorker)

    async def test_the_runner_is_run(self, wired):
        await main_module.main()
        assert FakeRunner.instances[0].ran == 1

    async def test_source_awaits_add_workers(self):
        """A non-awaited call would leave a dangling coroutine and connect to nothing."""
        source = inspect.getsource(main_module.run_pipeline)
        assert "await runner.add_workers(" in source


class TestSingleServiceRegression:
    """The gate must pause the service that is actually in the pipeline."""

    async def test_only_one_gemini_service_is_built(self, wired, monkeypatch):
        """Two services would leave the pipeline's copy paused forever.

        build_wake_gate receives the same GeminiLiveLLMService that goes
        into the pipeline. If run_pipeline built a second one, the gate
        would unpause an orphan while the pipeline's service stayed
        paused with start_audio_paused=True -- no audio would ever reach
        Google, the log would look healthy, and the plant would never
        answer. Same silent-failure family as the two regressions above.
        """
        built = []

        def counting_llm(**kwargs):
            built.append(kwargs)
            return "LLM"

        monkeypatch.setattr(main_module, "GeminiLiveLLMService", counting_llm)

        await main_module.main()

        assert len(built) == 1, f"run_pipeline built {len(built)} services, expected 1"


class TestPipelineOrder:
    async def test_full_processor_order(self, wired):
        await main_module.main()

        assert wired["processors"] == [
            "TRANSPORT_IN",
            "AGG_USER",
            "LLM",
            "TRANSPORT_OUT",
            "RECORDING_CONTROL",
            _audio_buffer(),
            "AGG_ASSISTANT",
        ]

    async def test_recording_control_sits_after_the_output_transport(self, wired):
        """It reacts to bot speaking frames, which originate downstream."""
        await main_module.main()
        processors = wired["processors"]
        assert processors.index("TRANSPORT_OUT") < processors.index("RECORDING_CONTROL")


class TestAudioConfiguration:
    async def test_transport_audio_params(self, wired):
        await main_module.main()

        params = wired["transport_params"]
        assert params.audio_in_enabled is True
        assert params.audio_in_channels == 1
        assert params.audio_in_sample_rate == 16000
        assert params.audio_out_enabled is True
        assert params.audio_out_channels == 1
        assert params.audio_out_sample_rate == 24000
        assert params.audio_out_10ms_chunks == 8


class TestLLMConfiguration:
    async def test_uses_the_settings_model_and_voice(self, wired):
        await main_module.main()

        settings = wired["llm_kwargs"]["settings"]
        assert settings.model == main_module.get_settings().gemini_live_model
        assert (
            settings.voice == main_module.get_settings().google_multimodal_live_voice_id
        )

    async def test_language_is_japanese(self, wired):
        await main_module.main()
        assert wired["llm_kwargs"]["settings"].language is main_module.Language.JA

    async def test_passes_the_system_instruction(self, wired):
        await main_module.main()
        assert (
            wired["llm_kwargs"]["system_instruction"] == main_module.SYSTEM_INSTRUCTION
        )

    async def test_does_not_use_deprecated_kwargs(self, wired):
        """model=/voice_id=/params= are removed in pipecat 2.0."""
        await main_module.main()
        kwargs = wired["llm_kwargs"]
        assert "model" not in kwargs
        assert "voice_id" not in kwargs
        assert "params" not in kwargs

    async def test_api_key_comes_from_settings(self, wired):
        await main_module.main()
        expected = main_module.get_settings().google_api_key.get_secret_value()
        assert wired["llm_kwargs"]["api_key"] == expected


class TestSystemInstruction:
    def test_names_the_persona(self):
        assert "PALM-9000" in main_module.SYSTEM_INSTRUCTION

    def test_requests_japanese_only_output(self):
        assert "日本語" in main_module.SYSTEM_INSTRUCTION


class TestHeartLifecycle:
    async def test_heart_is_started_and_stopped(self, wired):
        await main_module.main()

        heart = FakeHeart.instances[0]
        assert heart.started == 1
        assert heart.stopped == 1

    async def test_heart_stops_even_when_the_runner_raises(self, wired, monkeypatch):
        original = FakeRunner.run

        async def failing_run(self):
            self.ran += 1
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(FakeRunner, "run", failing_run)
        await main_module.main()  # must not propagate

        assert FakeHeart.instances[0].stopped == 1
        FakeRunner.run = original

    async def test_min_brightness_is_zero(self, wired):
        await main_module.main()
        assert FakeHeart.instances[0].kwargs["min_brightness"] == 0


class TestEventHandlers:
    async def test_audio_data_feeds_the_heart(self, wired):
        await main_module.main()

        buffer = _audio_buffer()
        handler = buffer.handlers["on_audio_data"][0]
        await handler(buffer, b"\x01\x02\x03\x04", 24000, 1)

        assert FakeHeart.instances[0].audio_chunks == [b"\x01\x02\x03\x04"]

    async def test_idle_timeout_handler_runs(self, wired):
        await main_module.main()

        worker = _worker()
        handler = worker.handlers["on_idle_timeout"][0]
        await handler(worker)  # logs only; must not raise

    async def test_idle_timeout_is_configured(self, wired):
        await main_module.main()

        worker = _worker()
        assert worker.kwargs["idle_timeout_secs"] == 600
        assert worker.kwargs["cancel_on_idle_timeout"] is True

    async def test_audio_buffer_size(self, wired):
        await main_module.main()
        assert _audio_buffer().kwargs["buffer_size"] == 512


# --- helpers ---------------------------------------------------------------


def _audio_buffer() -> FakeAudioBuffer:
    return FakeAudioBuffer.instances[0]


def _worker() -> FakeWorker:
    return FakeWorker.instances[0]


class TestHeartIsNeverLeftOn:
    """Regression: a failure during startup used to leave the matrix lit.

    heart.start() ran before the try/finally that stopped it, so anything
    raising in between -- a bad API key, a retired model id, no network --
    orphaned the render task with the display still on.
    """

    async def test_startup_failure_still_stops_the_heart(self, wired, monkeypatch):
        def exploding_service(**kwargs):
            raise RuntimeError("invalid api key")

        monkeypatch.setattr(main_module, "GeminiLiveLLMService", exploding_service)

        with pytest.raises(RuntimeError, match="invalid api key"):
            await main_module.main()

        heart = FakeHeart.instances[0]
        assert heart.started == 1
        assert heart.stopped == 1, "the display was left running after a failure"

    async def test_transport_failure_still_stops_the_heart(self, wired, monkeypatch):
        def exploding_transport(params):
            raise OSError("no audio device")

        monkeypatch.setattr(main_module, "LocalAudioTransport", exploding_transport)

        with pytest.raises(OSError):
            await main_module.main()

        assert FakeHeart.instances[0].stopped == 1

    async def test_heart_is_used_as_a_context_manager(self):
        """The guarantee is structural, not a try/finally someone can move."""
        source = inspect.getsource(main_module.main)
        assert "async with" in source
        assert "Max7219AmplitudeHeart" in source

    async def test_normal_run_still_stops_the_heart(self, wired):
        await main_module.main()
        heart = FakeHeart.instances[0]
        assert (heart.started, heart.stopped) == (1, 1)


class TestBuildersInIsolation:
    """The builders are testable without faking the whole module.

    Before the extraction, asserting anything about the pipeline meant
    monkeypatching eight module attributes and calling main().
    """

    def test_build_transport_audio_params(self, monkeypatch):
        captured = {}

        def fake_transport(params):
            captured["params"] = params
            return "TRANSPORT"

        monkeypatch.setattr(main_module, "LocalAudioTransport", fake_transport)

        assert main_module.build_transport() == "TRANSPORT"
        params = captured["params"]
        assert params.audio_in_sample_rate == main_module.AUDIO_IN_SAMPLE_RATE
        assert params.audio_out_sample_rate == main_module.AUDIO_OUT_SAMPLE_RATE
        assert params.audio_out_10ms_chunks == main_module.AUDIO_OUT_10MS_CHUNKS
        assert params.audio_in_enabled is True
        assert params.audio_out_enabled is True

    def test_build_llm_uses_settings_and_the_modern_kwargs(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "GeminiLiveLLMService",
            lambda **kw: captured.update(kw) or "LLM",
        )

        assert main_module.build_llm() == "LLM"
        assert "model" not in captured  # deprecated, removed in pipecat 2.0
        assert "voice_id" not in captured
        assert "params" not in captured
        assert (
            captured["settings"].model == main_module.get_settings().gemini_live_model
        )
        assert captured["settings"].language is main_module.Language.JA

    def test_build_pipeline_order(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )

        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
        )

        assert captured["p"] == [
            "IN",
            "AGG_USER",
            "LLM",
            "OUT",
            "CTL",
            "BUF",
            "AGG_ASSISTANT",
        ]

    def test_build_pipeline_keeps_the_user_aggregator_upstream_of_the_llm(
        self, monkeypatch
    ):
        """Regression guard: the realtime gate depends on this ordering."""
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )
        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
        )

        order = captured["p"]
        assert order.index("AGG_USER") < order.index("LLM")
        assert order.index("OUT") < order.index("CTL")


class TestNamedConstants:
    def test_audio_buffer_size_matches_the_documented_rate(self):
        """512 bytes at 24kHz mono int16 is ~10.7ms, so ~94 callbacks/sec."""
        ms = (
            main_module.AUDIO_BUFFER_SIZE / 2 / main_module.AUDIO_OUT_SAMPLE_RATE * 1000
        )
        assert 10 < ms < 11

    def test_idle_timeout_is_ten_minutes(self):
        assert main_module.IDLE_TIMEOUT_SECS == 600


class TestRunnerFailureKeepsTheTraceback:
    """logger.exception, not logger.error.

    Preserving the traceback is one of this PR's stated fixes, but the
    existing runner-failure test only checks that the heart stops -- swapping
    logger.exception back to logger.error leaves it green, so the fix could
    silently regress.
    """

    async def test_runner_failure_is_logged_with_its_traceback(
        self, wired, monkeypatch
    ):
        calls = {"exception": [], "error": []}
        monkeypatch.setattr(
            main_module.logger,
            "exception",
            lambda msg, *a, **kw: calls["exception"].append(msg),
        )
        monkeypatch.setattr(
            main_module.logger,
            "error",
            lambda msg, *a, **kw: calls["error"].append(msg),
        )

        async def failing_run(self):
            self.ran += 1
            raise RuntimeError("pipeline exploded")

        monkeypatch.setattr(FakeRunner, "run", failing_run)
        await main_module.main()

        assert calls["exception"], (
            "runner failure must be logged with logger.exception so the "
            "traceback survives"
        )
        assert "Pipeline error" in calls["exception"][0]
        assert not calls["error"], (
            "logger.error drops the traceback; use logger.exception"
        )


class TestWakeWordWiring:
    def test_no_gate_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=False)
        )
        assert main_module.build_wake_gate(llm=MagicMock()) is None

    def test_gate_built_when_enabled(self, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "get_settings",
            lambda: _settings(
                wake_word_enabled=True,
                wake_word_threshold=0.71,
                wake_silence_timeout_secs=17.5,
            ),
        )
        captured = {}
        monkeypatch.setattr(
            main_module, "WakeWordGate", lambda **kw: captured.update(kw) or "GATE"
        )

        class FakeDetector:
            def __init__(self, path, hop_samples=None):
                self.path = path
                self.hop_samples = hop_samples
                self.loaded = False

            def load(self):
                self.loaded = True

        monkeypatch.setattr(main_module, "LiveKitWakeWordDetector", FakeDetector)

        assert main_module.build_wake_gate(llm="LLM") == "GATE"
        assert captured["threshold"] == 0.71
        assert captured["silence_timeout_secs"] == 17.5
        assert captured["llm"] == "LLM"

    def test_hop_samples_reaches_the_detector(self, monkeypatch):
        """A non-default value, so hardcoding 1280 in build_wake_gate
        cannot pass this test."""
        monkeypatch.setattr(
            main_module,
            "get_settings",
            lambda: _settings(wake_word_enabled=True, wake_word_hop_samples=999),
        )
        monkeypatch.setattr(main_module, "WakeWordGate", lambda **kw: "GATE")

        captured = {}

        class FakeDetector:
            def __init__(self, path, hop_samples=None):
                captured["path"] = path
                captured["hop_samples"] = hop_samples

            def load(self):
                pass

        monkeypatch.setattr(main_module, "LiveKitWakeWordDetector", FakeDetector)
        main_module.build_wake_gate(llm="LLM")

        assert captured["hop_samples"] == 999

    def test_model_is_loaded_at_build_time(self, monkeypatch):
        """A missing model must fail at startup, not on the first frame."""
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        monkeypatch.setattr(main_module, "WakeWordGate", lambda **kw: "GATE")

        loaded = []

        class FakeDetector:
            def __init__(self, path, hop_samples=None):
                self.path = path

            def load(self):
                loaded.append(self.path)

        monkeypatch.setattr(main_module, "LiveKitWakeWordDetector", FakeDetector)
        main_module.build_wake_gate(llm="LLM")

        assert loaded == ["models/wakeword/hey_livekit.onnx"]

    def test_missing_model_raises_at_build_time(self, monkeypatch):
        monkeypatch.setattr(
            main_module,
            "get_settings",
            lambda: _settings(wake_word_enabled=True, wake_word_model_path="nope.onnx"),
        )
        with pytest.raises(FileNotFoundError):
            main_module.build_wake_gate(llm="LLM")

    def test_pipeline_places_the_gate_before_the_aggregator(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )
        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
            wake_gate="GATE",
        )

        order = captured["p"]
        assert order.index("IN") < order.index("GATE") < order.index("AGG_USER")

    def test_pipeline_omits_the_gate_when_none(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "Pipeline",
            lambda processors: captured.setdefault("p", processors),
        )
        transport = MagicMock()
        transport.input.return_value = "IN"
        transport.output.return_value = "OUT"
        aggregators = MagicMock()
        aggregators.user.return_value = "AGG_USER"
        aggregators.assistant.return_value = "AGG_ASSISTANT"

        main_module.build_pipeline(
            transport=transport,
            llm="LLM",
            context_aggregator=aggregators,
            recording_control="CTL",
            audio_buffer="BUF",
            wake_gate=None,
        )

        assert None not in captured["p"]
        assert captured["p"][0] == "IN"
        assert captured["p"][1] == "AGG_USER"

    def test_service_starts_paused_only_when_gating(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            main_module,
            "GeminiLiveLLMService",
            lambda **kw: captured.update(kw) or "LLM",
        )

        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        main_module.build_llm()
        assert captured["start_audio_paused"] is True

        captured.clear()
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=False)
        )
        main_module.build_llm()
        assert captured["start_audio_paused"] is False

    def test_aggregator_gets_a_vad_analyzer(self, monkeypatch):
        """Without it the silence timer has no activity signal."""
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=True)
        )
        captured = {}
        monkeypatch.setattr(
            main_module,
            "LLMContextAggregatorPair",
            lambda context, **kw: captured.update(kw) or MagicMock(),
        )
        main_module.build_context_aggregator()
        assert captured["user_params"].vad_analyzer is not None

    def test_aggregator_has_no_vad_analyzer_when_gating_is_off(self, monkeypatch):
        """The disabled path must stay identical to the pre-feature app.

        A vad_analyzer builds pipecat's VADController, which makes the
        default turn-start strategy broadcast interruptions. With no gate
        there is nothing that needs the signal, and the interruptions would
        let poor echo cancellation make the bot cut itself off.
        """
        monkeypatch.setattr(
            main_module, "get_settings", lambda: _settings(wake_word_enabled=False)
        )
        captured = {}
        monkeypatch.setattr(
            main_module,
            "LLMContextAggregatorPair",
            lambda context, **kw: captured.update(kw) or MagicMock(),
        )
        main_module.build_context_aggregator()
        assert captured == {}, "the disabled path must pass no user_params at all"
