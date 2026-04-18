#!/usr/bin/env python3
"""Transcribe medical audio using Google's MedASR (Conformer-CTC, 105M params)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before importing HF/torch — they read env vars (HF_TOKEN, HF_HOME, etc.)
# at import/config time, so ordering matters.
load_dotenv()

from transformers import pipeline  # noqa: E402

from clinical_ai.device import pick_device  # noqa: E402


def build_pipeline(model_id: str, device: str):
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
    )


def transcribe(
    audio_path: Path, model_id: str, device: str, chunk_s: float, stride_s: float
) -> str:
    pipe = build_pipeline(model_id, device)
    result = pipe(str(audio_path), chunk_length_s=chunk_s, stride_length_s=stride_s)
    return result["text"] if isinstance(result, dict) else str(result)


def fetch_sample(model_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_id, filename="test_audio.wav"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe medical audio with Google MedASR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("audio", type=Path, nargs="?", help="Path to an audio file (wav/mp3/flac/m4a).")
    ap.add_argument("--model", default="google/medasr", help="HF model id.")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device.",
    )
    ap.add_argument("--chunk-s", type=float, default=20.0, help="Chunk length in seconds.")
    ap.add_argument("--stride-s", type=float, default=2.0, help="Chunk overlap in seconds.")
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Download and transcribe the sample audio bundled with the model repo.",
    )
    args = ap.parse_args()

    if args.sample:
        audio_path = fetch_sample(args.model)
    elif args.audio is None:
        ap.error("audio path is required (or pass --sample)")
    else:
        audio_path = args.audio
        if not audio_path.exists():
            print(f"error: audio file not found: {audio_path}", file=sys.stderr)
            return 1

    device = pick_device() if args.device == "auto" else args.device
    print(f"[medasr] model={args.model} device={device} audio={audio_path}", file=sys.stderr)

    text = transcribe(audio_path, args.model, device, args.chunk_s, args.stride_s)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
