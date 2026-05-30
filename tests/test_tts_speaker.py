from __future__ import annotations

import time

import numpy as np

from src.config import AudioSettings, DEFAULT_CHATTERBOX_REF_WAV
from src.tts_speaker import TextSpeaker, split_tts_chunks


def test_split_tts_chunks_keeps_chatterbox_paragraphs():
    text = "First paragraph. Still first.\n\nSecond paragraph."

    assert split_tts_chunks(text, "chatterbox") == [
        "First paragraph. Still first.",
        "Second paragraph.",
    ]


def test_chatterbox_defaults_to_secondary_gpu(monkeypatch):
    monkeypatch.delenv("THERAPIST_CHATTERBOX_DEVICE", raising=False)

    assert AudioSettings().chatterbox_device == "cuda:1"


def test_chatterbox_defaults_to_repo_voice_sample(monkeypatch):
    monkeypatch.delenv("THERAPIST_CHATTERBOX_REF_WAV", raising=False)

    assert AudioSettings().chatterbox_ref_wav == str(DEFAULT_CHATTERBOX_REF_WAV)


def test_chatterbox_prefetches_next_chunk_during_playback(tmp_config, monkeypatch):
    speaker = TextSpeaker(tmp_config)
    speaker._backend = "chatterbox"

    first = "First paragraph."
    second = "Second paragraph."
    synth_started = {}
    play_finished = {}
    played = []
    chunk_ids = {first: 1.0, second: 2.0}
    id_to_chunk = {1.0: first, 2.0: second}

    monkeypatch.setattr(speaker, "_load_dependencies", lambda: True)

    def fake_synthesize(chunk):
        synth_started[chunk] = time.monotonic()
        time.sleep(0.05)
        return np.array([chunk_ids[chunk]], dtype=np.float32), 24000

    def fake_play(data, sample_rate):
        del sample_rate
        chunk = id_to_chunk[float(data[0])]
        time.sleep(0.15)
        play_finished[chunk] = time.monotonic()
        played.append(chunk)

    monkeypatch.setattr(speaker, "_synthesize_chunk", fake_synthesize)
    monkeypatch.setattr(speaker, "_play_audio", fake_play)

    speaker.speak_text(first + "\n\n" + second)

    assert played == [first, second]
    assert synth_started[second] < play_finished[first]


def test_chatterbox_cuda_fallback_uses_secondary_gpu(tmp_config, monkeypatch):
    speaker = TextSpeaker(tmp_config)
    speaker._backend = "chatterbox"
    speaker.config.audio.chatterbox_device = "cuda"
    monkeypatch.delenv("THERAPIST_CHATTERBOX_GPU_INDEX", raising=False)

    assert speaker._resolve_chatterbox_device() == "cuda:1"