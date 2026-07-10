# Batched low-VRAM path

This fork adds a fixed-batch decode path on top of upstream `FasterQwen3TTS`. It runs
N requests through a single set of `bs=N` CUDA graphs instead of one vLLM worker per
stream, which is what keeps VRAM low relative to `vLLM + Qwen3-TTS` for concurrent /
streaming workloads.

The single-stream and non-batched APIs are unchanged from upstream — you only opt into
batching explicitly via `enable_batch()`.

## Install (reproducible pin)

```bash
pip install "git+https://github.com/s4l4x/faster-qwen3-tts@v0.3.0-batch"
```

`v0.3.0-batch` is a tag off upstream `v0.3.0` plus the batching commits, so builds are
reproducible. Track `@batch-decode` instead only if you want the moving branch tip.

Requirements are the same as the base project: Python 3.10+, PyTorch 2.5.1+, an NVIDIA
GPU with CUDA. The batched path is **CUDA-only** — it captures fixed-batch CUDA graphs,
so there is no MPS/CPU fallback.

## What was added

- `FasterQwen3TTS.enable_batch(batch_size, max_seq_len=...)` — builds and captures the
  `bs=N` `PredictorGraph` / `TalkerGraph`. Call once after loading, before batched
  generation. Re-calling with the same `batch_size` is a no-op.
- `FasterQwen3TTS.generate_voice_clone_streaming_batch(requests)` — runs a list of
  requests as one fixed-batch generation and streams each slot's audio. Requires
  `enable_batch(bs)` first, where `bs == len(requests)`.
- `faster_qwen3_tts.streaming.fast_generate_streaming_batch(...)` — the lower-level
  batched decode loop.
- `examples/batch_server.py` — a micro-batching HTTP server built on the above.

## Minimal batched use

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", max_seq_len=768
)

# Warm the serial graphs once (prompt cache + capture) before capturing batched graphs.
for _ in model.generate_voice_clone_streaming(
    text="Warm up.", language="English",
    ref_audio="ref_audio.wav", ref_text="<transcript of ref_audio.wav>",
):
    pass

BS = 6
model.enable_batch(BS, max_seq_len=768)

requests = [
    {"text": "First line.",  "language": "English",
     "ref_audio": "ref_audio.wav", "ref_text": "<transcript>"},
    # ... exactly BS entries; pad short batches with a filler request (e.g. text=".")
    # since the batch size is fixed by the captured graphs.
]
for slot_idx, pcm_chunk in model.generate_voice_clone_streaming_batch(requests):
    ...  # route each slot's PCM16 audio to its caller
```

## Micro-batching server

`examples/batch_server.py` collects up to `BATCH_SIZE` requests within a
`COLLECT_WINDOW_MS` window, runs them as one batch, and streams PCM16 (mono, 24 kHz)
back per caller. Its HTTP shape matches the nano-server so existing benchmark scripts
work against it.

```bash
BATCH_SIZE=6 COLLECT_WINDOW_MS=50 PORT=8400 \
QWEN3_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
python examples/batch_server.py

# POST /v1/audio/speech {"text": ..., "language": ..., "speaker": ...}
#   -> chunked raw PCM16 mono 24 kHz
```

Configurable via env: `BATCH_SIZE`, `COLLECT_WINDOW_MS`, `MAX_SEQ_LEN`, `PORT`,
`QWEN3_TTS_MODEL`, `VOICE_REF_WAV`, `VOICE_REF_TEXT`.

## Notes / gotchas

- The batch size is **fixed at capture time**. `len(requests)` must equal the
  `batch_size` passed to `enable_batch`; pad short batches with a filler request that
  EOSes quickly (parked lanes are inert and cost ~nothing).
- Warm the serial (single-stream) graphs once before `enable_batch()` — the server does
  this with a tiny "Warm up." generation.
- Parity with the serial path is guarded by the layered bug-class discriminators in
  `tests/` rather than token-exact matching (batched sampling diverges bit-for-bit but
  must not exhibit the known failure classes).
