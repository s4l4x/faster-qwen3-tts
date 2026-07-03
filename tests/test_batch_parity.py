"""Batched-vs-serial parity for fast_generate_streaming_batch.

Greedy talker decode + greedy predictor graphs make both paths deterministic;
the remaining differences are batched-matmul FP reduction order. A mis-padded
or EOS-leaking slot diverges within the first few tokens, so the gate is the
first-divergence index, not token-exact equality.

Requires a CUDA GPU and downloads Qwen3-TTS-12Hz-1.7B-Base. Run manually:
    .venv/bin/python tests/test_batch_parity.py
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
REF_WAV = str(REPO / "ref_audio.wav")
REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish "
    "on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like "
    "learner, then this whole approach of training on verifiable outcomes is doomed."
)

TEXTS = [
    "I never told anyone this, but the summer I turned nineteen I drove all night.",
    "Sometimes I pretend I cannot hear my phone ringing at all.",
    "The dog was not lost. I just wanted one more day of walking around the neighborhood.",
]

GREEDY = dict(
    max_new_tokens=400,
    min_new_tokens=2,
    do_sample=False,
    repetition_penalty=1.05,
    chunk_size=12,
)


def build_greedy_graphs(model, batch_size):
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    m = model.model.model
    talker = m.talker
    talker_config = m.config.talker_config
    pg = PredictorGraph(
        talker.code_predictor,
        talker.code_predictor.model.config,
        talker_config.hidden_size,
        do_sample=False,
        batch_size=batch_size,
    )
    tg = TalkerGraph(talker.model, talker_config, max_seq_len=model.max_seq_len,
                     batch_size=batch_size)
    pg.capture(num_warmup=2)
    tg.capture(num_warmup=2)
    return pg, tg


def serial_tokens(model, pg, tg, text):
    from faster_qwen3_tts.streaming import fast_generate_streaming

    m, talker, config, tie, tam, tth, tpe, _ = model._prepare_generation(
        text=text, language="English", ref_audio=REF_WAV, ref_text=REF_TEXT,
        non_streaming_mode=False,
    )
    chunks = [
        chunk
        for chunk, _timing in fast_generate_streaming(
            talker=talker,
            talker_input_embeds=tie,
            attention_mask=tam,
            trailing_text_hiddens=tth,
            tts_pad_embed=tpe,
            config=config,
            predictor_graph=pg,
            talker_graph=tg,
            **GREEDY,
        )
    ]
    return torch.cat(chunks, dim=0)  # [T, 16]


def batch_tokens(model, pg, tg, texts):
    from faster_qwen3_tts.streaming import fast_generate_streaming_batch

    requests = [
        {"text": t, "language": "English", "ref_audio": REF_WAV, "ref_text": REF_TEXT}
        for t in texts
    ]
    m, talker, config, tie, tam, tth, tpe, _ = model._prepare_generation_batch(
        requests, non_streaming_mode=False
    )
    per_slot = [[] for _ in texts]
    for chunk, valid, _timing in fast_generate_streaming_batch(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=config,
        predictor_graph=pg,
        talker_graph=tg,
        **GREEDY,
    ):
        for b in range(len(texts)):
            rows = chunk[b][valid[b]]
            if rows.shape[0]:
                per_slot[b].append(rows)
    return [torch.cat(rows, dim=0) for rows in per_slot]  # each [T_b, 16]


def first_divergence(a: torch.Tensor, b: torch.Tensor) -> int:
    """Index of first differing frame (comparing all 16 codebooks); -1 if equal."""
    n = min(a.shape[0], b.shape[0])
    neq = (a[:n] != b[:n]).any(dim=1)
    idx = torch.nonzero(neq)
    if idx.numel() == 0:
        return -1 if a.shape[0] == b.shape[0] else n
    return int(idx[0])

def main():
    from faster_qwen3_tts import FasterQwen3TTS

    torch.manual_seed(0)
    model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    # Capture our own greedy graphs; skip the default stochastic warmup.
    bs = len(TEXTS)
    pg1, tg1 = build_greedy_graphs(model, 1)
    pgN, tgN = build_greedy_graphs(model, bs)
    model._warmed_up = True

    print("serial runs...")
    serial = [serial_tokens(model, pg1, tg1, t) for t in TEXTS]
    print("batched run...")
    batched = batch_tokens(model, pgN, tgN, TEXTS)

    ok = True
    for i, (s, b) in enumerate(zip(serial, batched)):
        div = first_divergence(s, b)
        frac = 1.0 if div < 0 else div / max(1, min(s.shape[0], b.shape[0]))
        status = "exact" if div < 0 else f"diverges at {div}/{s.shape[0]} vs {b.shape[0]}"
        print(f"slot {i}: serial {tuple(s.shape)} batched {tuple(b.shape)} -> {status}")
        # Gate: a padding/EOS bug diverges immediately; FP noise diverges late.
        if div >= 0 and frac < 0.4:
            ok = False
    if not ok:
        print("FAIL: early divergence (padding/EOS/state bug)")
        sys.exit(1)
    print("parity ok")


if __name__ == "__main__":
    main()
