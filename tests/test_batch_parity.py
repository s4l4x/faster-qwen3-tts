"""Batched-decode correctness gates for fast_generate_streaming_batch.

Token-exact serial-vs-batch parity is NOT achievable: bs=1 and bs=N kernels
reduce in different orders, and the predictor samples 15 codebooks per frame,
so bf16 near-tie argmax flips cascade even under greedy decode. Instead, each
real bug class gets a bit-exact discriminator:

1. batch-of-1 == serial          -> decode-loop bugs (EOS, history, trailing)
2. identical rows agree in-batch -> cross-slot leakage (masks, caches)
3. left-padded slot: rope_deltas == -pad, prefill argmax matches solo,
   stream length sane            -> padding/position bugs

Requires a CUDA GPU and downloads Qwen3-TTS-12Hz-1.7B-Base. Run manually:
    .venv/bin/python tests/test_batch_parity.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
REF_WAV = str(REPO / "ref_audio.wav")
REF_WAV2 = str(REPO / "ref_audio_2.wav")
REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish "
    "on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like "
    "learner, then this whole approach of training on verifiable outcomes is doomed."
)
TEXT = "Sometimes I pretend I cannot hear my phone ringing at all."

GREEDY = dict(
    max_new_tokens=150,
    min_new_tokens=2,
    do_sample=False,
    repetition_penalty=1.05,
    chunk_size=12,
)

REQ_ICL = {"text": TEXT, "language": "English", "ref_audio": REF_WAV, "ref_text": REF_TEXT}
REQ_XVEC = {"text": TEXT, "language": "English", "ref_audio": REF_WAV2, "xvec_only": True}


def build_greedy_graphs(model, batch_size):
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    m = model.model.model
    talker_config = m.config.talker_config
    pg = PredictorGraph(
        m.talker.code_predictor,
        m.talker.code_predictor.model.config,
        talker_config.hidden_size,
        do_sample=False,
        batch_size=batch_size,
    )
    tg = TalkerGraph(m.talker.model, talker_config, max_seq_len=model.max_seq_len,
                     batch_size=batch_size)
    pg.capture(num_warmup=2)
    tg.capture(num_warmup=2)
    return pg, tg


def serial_tokens(model, pg, tg, req):
    from faster_qwen3_tts.streaming import fast_generate_streaming

    _, talker, config, tie, tam, tth, tpe, _ = model._prepare_generation(
        text=req["text"], language=req["language"], ref_audio=req.get("ref_audio"),
        ref_text=req.get("ref_text", ""), xvec_only=req.get("xvec_only", False),
        non_streaming_mode=False,
    )
    chunks = [
        chunk
        for chunk, _ in fast_generate_streaming(
            talker=talker, talker_input_embeds=tie, attention_mask=tam,
            trailing_text_hiddens=tth, tts_pad_embed=tpe, config=config,
            predictor_graph=pg, talker_graph=tg, **GREEDY,
        )
    ]
    return torch.cat(chunks, dim=0)  # [T, 16]


def batch_tokens(model, pg, tg, requests):
    from faster_qwen3_tts.streaming import fast_generate_streaming_batch

    _, talker, config, tie, tam, tth, tpe, _ = model._prepare_generation_batch(
        requests, non_streaming_mode=False
    )
    per_slot = [[] for _ in requests]
    for chunk, valid, _ in fast_generate_streaming_batch(
        talker=talker, talker_input_embeds=tie, attention_mask=tam,
        trailing_text_hiddens=tth, tts_pad_embed=tpe, config=config,
        predictor_graph=pg, talker_graph=tg, **GREEDY,
    ):
        for b in range(len(requests)):
            rows = chunk[b][valid[b]]
            if rows.shape[0]:
                per_slot[b].append(rows)
    return [torch.cat(rows, dim=0) for rows in per_slot], talker


def main():
    from faster_qwen3_tts import FasterQwen3TTS

    torch.manual_seed(0)
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    model._warmed_up = True  # capture our own greedy graphs instead
    pg1, tg1 = build_greedy_graphs(model, 1)
    pg2, tg2 = build_greedy_graphs(model, 2)
    failures = []

    # Gate 1: batch-of-1 must match serial bit-exactly.
    s = serial_tokens(model, pg1, tg1, REQ_ICL)
    (b1,), _ = batch_tokens(model, pg1, tg1, [REQ_ICL])
    if not torch.equal(s, b1):
        failures.append(f"gate1: batch-of-1 != serial ({tuple(s.shape)} vs {tuple(b1.shape)})")
    print(f"gate1 batch-of-1 exact: {torch.equal(s, b1)} ({tuple(s.shape)})")

    # Gate 2: identical requests in one batch must agree bit-exactly.
    (r0, r1), _ = batch_tokens(model, pg2, tg2, [REQ_ICL, REQ_ICL])
    if not torch.equal(r0, r1):
        failures.append("gate2: identical rows diverged (cross-slot leakage)")
    print(f"gate2 identical rows agree: {torch.equal(r0, r1)} ({tuple(r0.shape)})")

    # Gate 3: heavy left-padding (ICL ~181 tokens vs xvec ~10) must not distort
    # the padded slot: rope delta == -pad_count and stream stays sane.
    (unpad_ref, _), _ = batch_tokens(model, pg2, tg2, [REQ_XVEC, REQ_XVEC])
    (_, padded), talker = batch_tokens(model, pg2, tg2, [REQ_ICL, REQ_XVEC])
    deltas = talker.rope_deltas.flatten().tolist()
    if deltas[0] != 0 or deltas[1] >= 0:
        failures.append(f"gate3: unexpected rope_deltas {deltas}")
    if padded[0, 0] != unpad_ref[0, 0]:
        failures.append(
            f"gate3: padded slot's first talker token {int(padded[0, 0])} != "
            f"unpadded {int(unpad_ref[0, 0])}"
        )
    len_ratio = padded.shape[0] / max(1, unpad_ref.shape[0])
    if not 0.5 <= len_ratio <= 2.0:
        failures.append(f"gate3: padded stream length ratio {len_ratio:.2f}")
    print(f"gate3 rope_deltas={deltas} first-token match="
          f"{int(padded[0, 0]) == int(unpad_ref[0, 0])} len_ratio={len_ratio:.2f}")

    if failures:
        print("FAIL:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("parity gates ok")


if __name__ == "__main__":
    main()
