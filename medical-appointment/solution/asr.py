"""Decode audio once and transcribe with two complementary timestamped Whisper models."""

import io
import time

MEDICAL_PROMPT = (
    "English primary-care medical consultation. Accurately transcribe medication names, "
    "doses, milligrams, micrograms, milliliters, dates, durations, body sides, lab values, "
    "vaccinations, examination findings, symptoms, treatment plans and follow-up instructions."
)


class _WhisperBackend:
    def __init__(self, config, model_path, beam_size, prompt=None):
        from faster_whisper import WhisperModel

        self.beam_size = beam_size
        self.prompt = prompt
        self.model = WhisperModel(
            model_path,
            device=config.device_type,
            device_index=config.device_index,
            compute_type=config.asr_compute_type,
            cpu_threads=config.cpu_threads,
            local_files_only=True,
        )

    def warmup(self):
        import numpy as np

        segments, _ = self.model.transcribe(
            np.zeros(16000, dtype=np.float32),
            language="en",
            vad_filter=False,
            beam_size=self.beam_size,
            word_timestamps=True,
        )
        list(segments)

    def transcribe_samples(self, samples):
        started = time.perf_counter()
        kwargs = dict(
            language="en",
            beam_size=self.beam_size,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        if self.prompt:
            kwargs["initial_prompt"] = self.prompt
        segments, _ = self.model.transcribe(samples, **kwargs)
        result = []
        for segment in segments:
            words = []
            for word in segment.words or []:
                if word.start is None or word.end is None:
                    continue
                words.append(
                    {"start": float(word.start), "end": float(word.end), "text": word.word}
                )
            result.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": segment.text.strip(),
                    "words": words,
                }
            )
        return result, time.perf_counter() - started


class Transcriber:
    """Returns a bundle with base and medium transcripts sharing the same audio clock."""

    def __init__(self, config):
        config.validate_device()
        self.config = config
        self.base = _WhisperBackend(
            config, config.asr_base_model, config.base_beam_size, prompt=None
        )
        self.medium = None
        if config.dual_asr:
            self.medium = _WhisperBackend(
                config,
                config.asr_medium_model,
                config.medium_beam_size,
                prompt=MEDICAL_PROMPT,
            )

    def warmup(self):
        self.base.warmup()
        if self.medium is not None:
            self.medium.warmup()

    def transcribe(self, audio_bytes):
        from faster_whisper.audio import decode_audio

        started = time.perf_counter()
        samples = decode_audio(io.BytesIO(audio_bytes), sampling_rate=16000)
        decode_time = time.perf_counter() - started
        duration = len(samples) / 16000.0

        base_segments, base_time = self.base.transcribe_samples(samples)
        sources = {
            "base": {"segments": base_segments, "duration": duration},
        }
        timings = {"decode": decode_time, "asr_base": base_time}

        if self.medium is not None:
            medium_segments, medium_time = self.medium.transcribe_samples(samples)
            sources["medium"] = {"segments": medium_segments, "duration": duration}
            timings["asr_medium"] = medium_time

        timings["asr"] = sum(v for k, v in timings.items() if k.startswith("asr_"))
        return {"sources": sources, "duration": duration, "timings": timings}
