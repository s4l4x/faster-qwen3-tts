"""Micro-batching streaming TTS server for the batched FasterQwen3TTS path.

Collects up to BATCH_SIZE requests inside a COLLECT_WINDOW_MS window, runs them
as one fixed-batch generation on the bs=N CUDA graphs, and streams each slot's
PCM16 audio back to its caller. Requests that arrive while a batch is running
wait for the next batch (worst case: one full generation).

API matches the nano-server shape so scripts/ramp_benchmark.py --api nano works:
  POST /v1/audio/speech {"text": ..., "language": ..., "speaker": ...}
  -> chunked raw PCM16 mono 24 kHz

Env:
  BATCH_SIZE (6), COLLECT_WINDOW_MS (50), PORT (8400),
  QWEN3_TTS_MODEL (Qwen/Qwen3-TTS-12Hz-1.7B-Base), MAX_SEQ_LEN (768),
  VOICE_REF_WAV / VOICE_REF_TEXT (defaults: repo ref_audio.wav + its transcript).
"""
import asyncio
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_server")

REPO = Path(__file__).resolve().parent.parent
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "6"))
COLLECT_WINDOW_MS = int(os.environ.get("COLLECT_WINDOW_MS", "50"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "768"))
MODEL_NAME = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE_REF_WAV = os.environ.get("VOICE_REF_WAV", str(REPO / "ref_audio.wav"))
VOICE_REF_TEXT = os.environ.get(
    "VOICE_REF_TEXT",
    "I'm confused why some people have super short timelines, yet at the same time are "
    "bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a "
    "human-like learner, then this whole approach of training on verifiable outcomes is doomed.",
)
SAMPLE_RATE = 24000

_model = None
_pending: asyncio.Queue = None
_loop: asyncio.AbstractEventLoop = None
_END = object()


class Job:
    def __init__(self, text: str, language: str):
        self.text = text
        self.language = language
        self.out: asyncio.Queue = asyncio.Queue()
        self.queued_at = time.time()


def _load_model():
    global _model
    from faster_qwen3_tts import FasterQwen3TTS

    logger.info(f"loading {MODEL_NAME} ...")
    _model = FasterQwen3TTS.from_pretrained(MODEL_NAME, max_seq_len=MAX_SEQ_LEN)
    # Warm the serial graphs (prompt cache + capture) with a tiny generation,
    # then capture the batched graphs.
    for _ in _model.generate_voice_clone_streaming(
        text="Warm up.", language="English",
        ref_audio=VOICE_REF_WAV, ref_text=VOICE_REF_TEXT,
    ):
        pass
    _model.enable_batch(BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    logger.info(f"model ready (bs={BATCH_SIZE}, max_seq_len={MAX_SEQ_LEN})")


def _run_batch(jobs: list):
    """Run one fixed-batch generation on the GPU thread; route audio to jobs."""
    requests = [
        {"text": j.text, "language": j.language,
         "ref_audio": VOICE_REF_WAV, "ref_text": VOICE_REF_TEXT}
        for j in jobs
    ]
    # Pad the batch to BATCH_SIZE with a filler that EOSes almost immediately;
    # parked lanes are inert, so filler costs nothing beyond the fixed batch.
    n_real = len(requests)
    while len(requests) < BATCH_SIZE:
        requests.append({"text": ".", "language": "English",
                         "ref_audio": VOICE_REF_WAV, "ref_text": VOICE_REF_TEXT})

    t0 = time.time()
    try:
        for slot, audio, sr, _timing in _model.generate_voice_clone_streaming_batch(requests):
            if slot >= n_real:
                continue
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            _loop.call_soon_threadsafe(jobs[slot].out.put_nowait, pcm16)
    except Exception as exc:  # surface errors to all callers rather than hanging them
        logger.exception("batch generation failed")
        for j in jobs:
            _loop.call_soon_threadsafe(j.out.put_nowait, exc)
    finally:
        for j in jobs:
            _loop.call_soon_threadsafe(j.out.put_nowait, _END)
        logger.info(f"batch of {n_real} done in {time.time() - t0:.2f}s")


async def _scheduler():
    """Collect jobs into batches; one batch on the GPU at a time."""
    while True:
        first = await _pending.get()
        jobs = [first]
        deadline = time.time() + COLLECT_WINDOW_MS / 1000
        while len(jobs) < BATCH_SIZE:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(_pending.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        waits = [f"{time.time() - j.queued_at:.2f}" for j in jobs]
        logger.info(f"dispatching batch of {len(jobs)} (queue waits: {waits}s)")
        await asyncio.to_thread(_run_batch, jobs)


app = FastAPI(title="FasterQwen3TTS batch server")


@app.on_event("startup")
async def startup():
    global _pending, _loop
    _loop = asyncio.get_running_loop()
    _pending = asyncio.Queue()
    await asyncio.to_thread(_load_model)
    asyncio.create_task(_scheduler())


@app.get("/health")
async def health():
    return {"status": "ok", "batch_size": BATCH_SIZE}


@app.post("/v1/audio/speech")
async def speech(body: dict):
    job = Job(text=body["text"], language=body.get("language", "English"))
    await _pending.put(job)

    async def stream():
        while True:
            item = await job.out.get()
            if item is _END:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    return StreamingResponse(
        stream(), media_type="audio/L16", headers={"Sample-Rate": str(SAMPLE_RATE)}
    )


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8400")))
